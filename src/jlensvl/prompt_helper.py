"""Forward-only prompt diagnostics via the J-Lens.

Idea: the J-Lens reads what a model is *poised to say* at the end of a prompt,
before it generates. So you can (a) see the concept a prompt steers toward and
how decisive it is, and (b) rank alternative phrasings by how strongly they push
the model to an *intended* sense versus competing senses.

This is a diagnostic/comparison aid (white-box; needs the model + a fitted lens),
not an automatic prompt generator. Every call is a single forward pass.
"""
from __future__ import annotations
import html


class PromptHelper:
    def __init__(self, jl):
        """jl: a fitted JLensVL instance."""
        self.jl = jl

    def _layer(self, layer):
        if layer is not None:
            return layer
        sl = self.jl.lens.source_layers
        # penultimate source layer, but fall back to the only layer for a
        # single-source-layer lens (source_layers[-2] would IndexError).
        return sl[-2] if len(sl) >= 2 else sl[-1]

    def poised(self, prompt, *, layer=None, k=6):
        """What is the model poised to say at the end of `prompt`?
        Returns top-k tokens + a decisiveness margin (top1 - top2)."""
        jl = self.jl; jl._require_lens(); L = self._layer(layer)
        ll, _, _ = jl.lens.apply(jl.lm, prompt, positions=[-1], layers=[L], use_jacobian=True)
        vals, idx = ll[L][0].topk(k)
        toks = jl._decode(idx.tolist())
        margin = float(vals[0] - vals[1]) if vals.numel() > 1 else float("inf")
        return {"tokens": toks, "top1": toks[0], "margin": margin, "layer": L}

    def poised_continuation(self, messages, prefill, *, reasoning=False, layer=None,
                            senses=None, k=6):
        """Counterfactual assistant-prefill: *if the model had already committed
        to `prefill`, what is it poised to say NEXT?*

        Renders `messages` with an extra trailing **assistant** turn carrying the
        prefill and uses ``continue_final_message=True`` (NOT
        ``add_generation_prompt`` — the two are mutually exclusive) so the model
        continues that same turn rather than opening a fresh one. `prefill` goes
        in as the partial answer ``content``, or as partial chain-of-thought
        ``reasoning_content`` when ``reasoning=True``.

        Caveat: templates that close ``</think>`` around ``reasoning_content``
        (ours does) leave the model poised right AFTER the reasoning block — i.e.
        "given this reasoning, what's the answer" — not mid-thought.

        Returns top-k poised tokens + decisiveness margin (and optional
        `sense_scores` when `senses` is given).
        """
        jl = self.jl; jl._require_lens(); tok = jl.tok; L = self._layer(layer)
        turn = {"role": "assistant"}
        turn["reasoning_content" if reasoning else "content"] = prefill
        msgs = list(messages) + [turn]
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, continue_final_message=True,
            add_generation_prompt=False)
        ll, _, _ = jl.lens.apply(jl.lm, rendered, positions=[-1], layers=[L],
                                 use_jacobian=True)
        logits = ll[L][0]
        vals, idx = logits.topk(k)
        toks = jl._decode(idx.tolist())
        margin = float(vals[0] - vals[1]) if vals.numel() > 1 else float("inf")
        out = {"tokens": toks, "top1": toks[0], "margin": margin,
               "layer": int(L), "reasoning": bool(reasoning), "rendered": rendered}
        if senses:
            out["senses"] = {n: self._score_words(logits, w)
                             for n, w in senses.items()}
        return out

    def _score_words(self, logits, words):
        """Max poised logit over `words`' token ids, with the `_phrase_ids`
        fallback so unknown / multi-token senses don't crash on an empty max()."""
        ids = self.jl._word_ids(words)
        if not ids:
            acc = set()
            for w in words:
                acc.update(self._phrase_ids(w))
            ids = sorted(acc)
        return float(logits[ids].max()) if ids else float("-inf")

    @staticmethod
    def _tool_names(tools):
        """Extract callable/schema tool names from an OpenAI-style `tools` list."""
        names = []
        for t in tools or []:
            if callable(t):
                names.append(getattr(t, "__name__", None))
                continue
            fn = t.get("function", t) if isinstance(t, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                names.append(fn["name"])
        return [n for n in names if n]

    def _phrase_ids(self, text):
        """Single-token ids for `text` (reusing `_word_ids`), else the first
        sub-token of each surface form — so multi-token tool/arg names still get
        a defined poised score instead of an empty max()."""
        ids = self.jl._word_ids([text])
        if ids:
            return ids
        unk = getattr(self.jl.tok, "unk_token_id", None)
        out = set()
        for v in (text, " " + text):
            e = self.jl.tok.encode(v, add_special_tokens=False)
            if e and e[0] != unk:
                out.add(int(e[0]))
        return sorted(out)

    def poised_tool(self, messages, tools, *, layer=None, tool_names=None,
                    arg_names=None, enable_thinking=None):
        """Which tool / arg is the model poised toward at the generation point?

        Renders the tool-calling prompt (threading `tools` through the template)
        and scores each candidate tool name (and optional `arg_names`) with the
        same J-Lens readout as `sense_scores`: the max poised logit of the name's
        token ids at the answer position. No-op (returns ``{}``) when `tools` is
        falsy. `tool_names` defaults to the names parsed from `tools`.
        """
        jl = self.jl; jl._require_lens(); L = self._layer(layer)
        if not tools:
            return {}
        names = list(tool_names) if tool_names is not None else self._tool_names(tools)
        rendered = self._render(messages, add_generation_prompt=True,
                                enable_thinking=enable_thinking, tools=tools)
        ll, _, _ = jl.lens.apply(jl.lm, rendered, positions=[-1], layers=[L],
                                 use_jacobian=True)
        logits = ll[L][0]

        def score(name):
            ids = self._phrase_ids(name)
            return float(logits[ids].max()) if ids else float("-inf")

        tool_scores = {n: score(n) for n in names}
        out = {"layer": int(L), "tools": tool_scores,
               "top_tool": max(tool_scores, key=tool_scores.get) if tool_scores else None}
        if arg_names:
            out["args"] = {a: score(a) for a in arg_names}
            if out["args"]:
                out["top_arg"] = max(out["args"], key=out["args"].get)
        return out

    def sense_scores(self, prompt, senses, *, layer=None):
        """J-Lens score of each candidate sense at the answer position.
        `senses` = {name: [words]}. Higher = the model leans that way."""
        jl = self.jl; jl._require_lens(); L = self._layer(layer)
        ll, _, _ = jl.lens.apply(jl.lm, prompt, positions=[-1], layers=[L], use_jacobian=True)
        logits = ll[L][0]
        return {n: self._score_words(logits, w) for n, w in senses.items()}

    def rank_prompts(self, variants, senses, intended, *, layer=None):
        """Rank prompt `variants` by how well they steer to the `intended` sense
        versus the strongest competing sense.

        Args:
            variants: list of prompts, or {name: prompt}.
            senses: {sense_name: [words]} candidate meanings.
            intended: the sense_name you want the prompt to elicit.
        Returns: list of dicts sorted best-first by (intended - best_competitor).
        """
        items = variants.items() if isinstance(variants, dict) else [(p, p) for p in variants]
        rows = []
        for name, prompt in items:
            sc = self.sense_scores(prompt, senses, layer=layer)
            comp = max((v for k, v in sc.items() if k != intended), default=float("-inf"))
            rows.append({"name": name, "prompt": prompt, "scores": sc,
                         "intended": sc.get(intended), "best_competitor": comp,
                         "margin": sc.get(intended, float("-inf")) - comp})
        rows.sort(key=lambda r: r["margin"], reverse=True)
        return rows

    def report(self, variants, senses, intended, *, layer=None, width=22):
        """Human-readable, visual ranking of prompt variants: ASCII bars of each
        sense's J-Lens score, a per-prompt verdict, and the overall winner.
        Returns a printable string."""
        rows = self.rank_prompts(variants, senses, intended, layer=layer)
        allv = [v for r in rows for v in r["scores"].values() if v is not None]
        mx = max(allv) if allv else 1.0
        out = [f"Prompt-Helper report — intended sense: {intended!r}  "
               f"(J-Lens @ layer {self._layer(layer)})", "=" * 68]
        for rank, r in enumerate(rows, 1):
            m = r["margin"]
            verdict = "✓ CLEAR" if m >= 3 else ("~ weak/ambiguous" if m > 0 else "✗ OFF-TARGET")
            out.append(f"\n#{rank}  [{r['name']}]  {r['prompt']!r}")
            for s, v in sorted(r["scores"].items(), key=lambda x: -(x[1] or -1e9)):
                nb = int(round(width * max(v or 0, 0) / mx)) if mx > 0 else 0
                tag = "  ← intended" if s == intended else ""
                out.append(f"    {s:12s} |{'█' * nb:<{width}}| {v:6.2f}{tag}")
            out.append(f"    → margin (intended − best competitor) = {m:+.2f}   [{verdict}]")
        out.append(f"\nRANKING (best first): {' > '.join(r['name'] for r in rows)}")
        b = rows[0]
        out.append(f"VERDICT: use [{b['name']}] — it steers the model to {intended!r} "
                   f"with a {b['margin']:+.2f} margin over the next sense.")
        return "\n".join(out)

    # ---------- template-aware (chat_template.jinja) ----------
    @staticmethod
    def _is_unsupported_kwarg_error(exc: TypeError, name: str) -> bool:
        """True only for the CANONICAL "unexpected keyword argument" signature
        error about `name` (e.g. transformers predating the flag) — NOT some
        other TypeError raised from inside the template that merely mentions
        `name`. Kept deliberately narrow so a genuine request is never dropped."""
        msg = str(exc)
        return name in msg and ("unexpected keyword argument" in msg
                                or "got an unexpected keyword" in msg)

    def _thinking_flag_effective(self):
        """Cached: does passing `enable_thinking` actually CHANGE what the
        template renders? ``False`` means the flag is accepted-but-ignored (or
        unsupported at the signature level) — so an ``enable_thinking=False``
        request cannot really be honored. ``None`` when undeterminable."""
        cached = getattr(self, "_think_eff_cache", "unset")
        if cached != "unset":
            return cached
        tok = self.jl.tok
        probe = [{"role": "user", "content": "x"}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            on = tok.apply_chat_template(probe, enable_thinking=True, **kw)
            off = tok.apply_chat_template(probe, enable_thinking=False, **kw)
            eff = (on != off)
        except TypeError as exc:
            # Canonical "unexpected keyword" => flag genuinely unsupported.
            # Any OTHER TypeError is undeterminable — don't misread it as
            # "ineffective" (that would mask a real template bug).
            eff = False if self._is_unsupported_kwarg_error(exc, "enable_thinking") else None
        except Exception:  # noqa: BLE001 - undeterminable, don't block
            eff = None
        self._think_eff_cache = eff
        return eff

    def _thinking_default(self):
        """Best-effort: does THIS chat template default thinking ON?

        Portability note: our 4B template defaults ON, but sibling templates
        (Qwen3.5-2B/9B) default OFF. Rather than assume, we render a trivial
        generation prompt with no `enable_thinking` and compare it against the
        explicit `enable_thinking=False` rendering. If they differ, the implicit
        default is ON. Returns True/False, or None when the flag is ineffective
        (unsupported OR accepted-but-ignored) so "thinking default" is moot."""
        if self._thinking_flag_effective() is not True:
            return None
        tok = self.jl.tok
        probe = [{"role": "user", "content": "x"}]
        default = tok.apply_chat_template(probe, tokenize=False,
                                          add_generation_prompt=True)
        off = tok.apply_chat_template(probe, tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        return default != off

    def _render(self, messages, *, add_generation_prompt=True, enable_thinking=None,
                tools=None, add_vision_id=None):
        """messages -> the exact chat-template-rendered string the model sees.

        Universal `apply_chat_template` passthroughs:
          * `enable_thinking` (None = the template's own default; see
            `_thinking_default`). A genuine `enable_thinking=False` is NEVER
            silently dropped — if the tokenizer truly can't honor it we raise.
          * `tools` — tool schemas for tool-calling prompts (no-op when None).
          * `add_vision_id` — Qwen VLM image tagging (no-op / ignored when the
            template doesn't reference it).
        """
        tok = self.jl.tok
        kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        if tools is not None:
            kw["tools"] = tools
        if add_vision_id is not None:
            kw["add_vision_id"] = add_vision_id
        if enable_thinking is not None:
            # An explicit `enable_thinking=False` we cannot actually effect (the
            # template ignores or rejects the flag) must NOT be silently honored
            # in name only — surface it rather than render thinking-on regardless.
            if enable_thinking is False and self._thinking_flag_effective() is False:
                raise TypeError(
                    "enable_thinking=False was requested but this tokenizer's "
                    "apply_chat_template ignores/rejects the flag (identical render "
                    "with and without it); refusing to silently render with thinking "
                    "possibly enabled.")
            try:
                return tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kw)
            except TypeError as exc:
                # Only fall back when `enable_thinking` is genuinely unsupported
                # by the signature — never swallow an unrelated TypeError.
                if not self._is_unsupported_kwarg_error(exc, "enable_thinking"):
                    raise
                # enable_thinking=True but unsupported -> template has no thinking
                # concept; fall through to the plain render. (False already
                # raised above via the effectiveness guard.)
        return tok.apply_chat_template(messages, **kw)

    @staticmethod
    def _phantom_think_positions(toks):
        """Positions belonging to a PHANTOM empty ``<think>\\n\\n</think>`` block
        (the one the template injects when thinking is OFF). Detected as a
        ``<think>`` token followed — across whitespace-only tokens — immediately
        by ``</think>`` with no real content between. Returned as a set so callers
        can skip it; leaving it in skews per-token traces / diagnose_thinking on
        multi-turn and tool prompts."""
        phantom = set()
        n = len(toks)
        for p, t in enumerate(toks):
            if t != "<think>":
                continue
            run = [p]
            q = p + 1
            while q < n and toks[q].strip() == "":
                run.append(q)
                q += 1
            if q < n and toks[q] == "</think>":
                run.append(q)
                phantom.update(run)
        return phantom

    def trace_rendered(self, messages, *, senses=None, enable_thinking=None,
                       add_generation_prompt=True, tools=None, add_vision_id=None,
                       layer=None, topk=6):
        """Run the J-Lens on the REAL chat-template-rendered token sequence (not the
        raw string). Returns a per-token trace + segment/special tags + the sense
        scores at the answer position. This is the faithful thing to lens for a
        chat model — the model never sees your raw text, it sees the rendered tokens.

        `tools` / `add_vision_id` are universal `apply_chat_template` passthroughs
        (no-op when None) so tool-calling / VLM prompts can be lensed faithfully.

        Answer-position caveat: `answer = len(ids) - 1` is the FINAL rendered
        token, which differs by thinking mode — thinking-on ends at an open
        ``<think>\\n`` (poised to reason), thinking-off ends past the closed
        ``</think>\\n\\n`` block (poised to answer). So sense scores across modes
        are read at genuinely different residual states; that's intended, but
        don't treat the two positions as identical.
        """
        jl = self.jl; jl._require_lens()
        tok = jl.tok
        rendered = self._render(messages, add_generation_prompt=add_generation_prompt,
                                enable_thinking=enable_thinking, tools=tools,
                                add_vision_id=add_vision_id)
        layers = jl.lens.source_layers
        ll, ml, ids = jl.lens.apply(jl.lm, rendered, positions=None, layers=layers, use_jacobian=True)
        if hasattr(ids, "dim") and ids.dim() > 1:
            ids = ids[0]
        ids = [int(i) for i in ids]
        toks = [tok.decode([i]) for i in ids]
        L = layer if layer is not None else layers[-2]
        special = set(getattr(tok, "all_special_ids", []) or [])
        phantom = self._phantom_think_positions(toks)
        role = None
        per = []
        for p, i in enumerate(ids):
            surf = toks[p]
            is_sp = i in special or ("<|" in surf and "|>" in surf) or surf in ("<think>", "</think>")
            if "im_start" in surf:
                role = "?"
            elif role == "?" and surf.strip() in ("system", "user", "assistant", "tool"):
                role = surf.strip()
            elif "im_end" in surf:
                role = None
            idx = [int(j) for j in ll[L][p].topk(topk).indices.tolist()]
            per.append({"tok": surf, "top": [tok.decode([j]) for j in idx],
                        "special": is_sp or (p in phantom), "role": role,
                        "phantom": p in phantom})
        ans = len(ids) - 1
        sc = None
        if senses:
            alog = ll[L][ans]
            sc = {n: self._score_words(alog, w) for n, w in senses.items()}
        return {"rendered": rendered, "tokens": toks, "per": per, "answer": ans,
                "layer": int(L), "senses": sc, "phantom_think": sorted(phantom)}

    def compare_templates(self, base_messages, variants, senses, intended, *, layer=None):
        """A/B different template configs (system on/off, thinking on/off, few-shot…)
        by how strongly each steers to `intended` sense at the answer position.
        `variants` = {name: {"messages": …?, "enable_thinking": …?}}. Best-first list.
        """
        rows = []
        for name, cfg in variants.items():
            tr = self.trace_rendered(cfg.get("messages", base_messages), senses=senses,
                                     enable_thinking=cfg.get("enable_thinking"), layer=layer)
            sc = tr["senses"]
            comp = max((v for k, v in sc.items() if k != intended), default=float("-inf"))
            rows.append({"name": name, "scores": sc, "intended": sc.get(intended),
                         "best_competitor": comp, "margin": sc.get(intended, float("-inf")) - comp})
        rows.sort(key=lambda r: r["margin"], reverse=True)
        return rows

    def check_system_registers(self, messages, senses, intended, *, layer=None):
        """Does the system message actually 'land'? Compare the intended-sense score
        at the answer position WITH vs WITHOUT the system message."""
        with_sys = self.trace_rendered(messages, senses=senses, layer=layer)["senses"][intended]
        no_sys = [m for m in messages if m.get("role") != "system"]
        without = self.trace_rendered(no_sys, senses=senses, layer=layer)["senses"][intended]
        return {"with_system": with_sys, "without_system": without,
                "delta": with_sys - without,
                "verdict": "registers" if abs(with_sys - without) >= 1.0 else "no measurable effect"}

    def diagnose_thinking(self, messages, senses, intended, *, layer=None):
        """Does enabling Qwen's thinking mode help or hurt the intended sense at the
        answer position? Compares trace_rendered with enable_thinking=True vs =False.
        Returns a dict: intended-sense score and margin(intended - best competitor) for
        each mode, their deltas, and a verdict ('helps' / 'hurts' / 'no measurable effect')."""
        def _mode(enable_thinking):
            sc = self.trace_rendered(messages, senses=senses,
                                     enable_thinking=enable_thinking, layer=layer)["senses"]
            if sc is None:
                raise ValueError("trace_rendered returned no senses; pass a non-empty `senses`.")
            intended_score = sc[intended]
            comp = max((v for k, v in sc.items() if k != intended), default=float("-inf"))
            return {"intended": intended_score, "margin": intended_score - comp}
        on = _mode(True)
        off = _mode(False)
        delta_margin = on["margin"] - off["margin"]
        verdict = ("helps" if delta_margin >= 1.0
                   else "hurts" if delta_margin <= -1.0 else "no measurable effect")
        return {"thinking_on": on, "thinking_off": off,
                "delta_margin": delta_margin, "verdict": verdict}

    # ---------- self-contained HTML report ----------
    def report_html(self, base_messages, variants, senses, intended, *, layer=None,
                    out_path=None, title="JLensVL prompt-helper report"):
        """A single self-contained HTML prompt-helper report (inline CSS/JS, no
        external deps, dark/light, offline).

        Two parts:
        1. A grouped horizontal-bar ranking of the template `variants`
           (`compare_templates`): per variant, one bar per sense (width ∝ score;
           the `intended` sense highlighted green, others muted), the margin, and a
           verdict (✓ CLEAR / ~ weak / ✗ OFF-TARGET). Best-first.
        2. A token strip over the REAL chat-template-rendered sequence of the
           winning variant (`trace_rendered`) — mirrors `viz.rendered_strip_html`'s
           look: special tokens blue, answer position outlined orange, hover shows
           the poised concept.

        Reuses `viz._HEAD` for head/theme so styling matches the rest of JLensVL.
        Writes to `out_path` if given; always returns the HTML string.
        """
        from . import viz  # inside method to avoid any import-order surprises

        rows = self.compare_templates(base_messages, variants, senses, intended, layer=layer)

        def esc(s):
            return html.escape(str(s))

        def verdict(m):
            return "✓ CLEAR" if m >= 3 else ("~ weak" if m > 0 else "✗ OFF-TARGET")

        # --- part 1: grouped horizontal bar ranking ---
        allv = [v for r in rows for v in r["scores"].values() if v is not None]
        vmax = max(allv) if allv else 1e-6
        vmax = vmax if vmax > 0 else 1e-6
        rank_html = ""
        for i, r in enumerate(rows, 1):
            m = r["margin"] if r["margin"] is not None else float("-inf")
            vd = verdict(m)
            vcls = "vok" if m >= 3 else ("vweak" if m > 0 else "voff")
            bars = ""
            for s, v in sorted(r["scores"].items(), key=lambda x: -(x[1] if x[1] is not None else -1e9)):
                val = v if v is not None else 0.0
                w = max(2, int(round(max(val, 0.0) / vmax * 240)))
                hl = "background:var(--gn)" if s == intended else "background:var(--mu)"
                tag = " ← intended" if s == intended else ""
                bars += (f'<div class="bar">'
                         f'<span class="bn">{esc(s)}{tag}</span>'
                         f'<span class="bt" style="width:{w}px;{hl}"></span>'
                         f'<span class="bv">{val:.2f}</span></div>')
            mtxt = f"{m:+.2f}" if m != float("-inf") else "n/a"
            rank_html += (f'<div class="var">'
                          f'<div class="vh"><span class="rk">#{i}</span>'
                          f'<span class="vn">{esc(r["name"])}</span>'
                          f'<span class="vd {vcls}">{esc(vd)}</span></div>'
                          f'{bars}'
                          f'<div class="mg">margin (intended − best competitor) = {esc(mtxt)}</div>'
                          f'</div>')

        # --- part 2: token strip of the winning variant ---
        win = rows[0]["name"]
        win_cfg = variants[win]
        win_messages = win_cfg.get("messages", base_messages)
        win_thinking = win_cfg.get("enable_thinking")
        trace = self.trace_rendered(win_messages, senses=senses,
                                    enable_thinking=win_thinking, layer=layer)
        boxes = ""
        for p, c in enumerate(trace["per"]):
            surf = c["tok"].replace("\n", "\\n") or "·"
            tip = esc("L%d 欲言: %s" % (trace["layer"], " · ".join(c["top"])))
            cls = "sp" if c["special"] else "ct"
            if p == trace["answer"]:
                cls += " ans"
            role = c.get("role")
            rl = f' data-role="{esc(role)}"' if role else ""
            boxes += f'<span class="tk {cls}"{rl} title="{tip}">{esc(surf)}</span>'

        # --- diagnostic callout cards for the winning variant ---
        diag_html = ""

        # 1. system-registers card (only if the winning messages have a system role)
        if any(m.get("role") == "system" for m in win_messages):
            sr = self.check_system_registers(win_messages, senses, intended, layer=layer)
            sr_pill = "pill-gn" if sr["verdict"] == "registers" else "pill-or"
            diag_html += (
                '<div class="diag-card">'
                '<div class="dt">System 消息是否「落地」？（with vs without system）</div>'
                '<div class="dn">'
                f'<span><b>with_system:</b> {sr["with_system"]:.2f}</span>'
                f'<span><b>without_system:</b> {sr["without_system"]:.2f}</span>'
                f'<span><b>delta:</b> {sr["delta"]:+.2f}</span>'
                f'<span class="pill {sr_pill}">{esc(sr["verdict"])}</span>'
                '</div></div>')

        # 2. thinking card (guarded: some tokenizers/paths don't support thinking mode)
        try:
            dt = self.diagnose_thinking(win_messages, senses, intended, layer=layer)
        except Exception:
            dt = None
        if dt is not None:
            dv = dt["verdict"]
            dt_pill = ("pill-gn" if dv == "helps"
                       else "pill-or" if dv == "hurts" else "pill-mu")
            diag_html += (
                '<div class="diag-card">'
                '<div class="dt">Thinking 模式是否有帮助？（thinking on vs off）</div>'
                '<div class="dn">'
                f'<span><b>on margin:</b> {dt["thinking_on"]["margin"]:+.2f}</span>'
                f'<span><b>off margin:</b> {dt["thinking_off"]["margin"]:+.2f}</span>'
                f'<span><b>delta_margin:</b> {dt["delta_margin"]:+.2f}</span>'
                f'<span class="pill {dt_pill}">{esc(dv)}</span>'
                '</div></div>')

        diag_section = (f'<h3 style="margin:1.4em 0 .3em;color:var(--ac)">诊断卡片 · [{esc(win)}]</h3>'
                        f'{diag_html}') if diag_html else ""

        # --- extra CSS (bars + token strip, mirroring rendered_strip_html) ---
        css = (
            '.var{border:1px solid var(--bd);border-radius:8px;background:var(--pan);'
            'padding:10px 12px;margin:10px 0}'
            '.vh{display:flex;align-items:center;gap:10px;margin-bottom:6px}'
            '.rk{color:var(--mu);font:700 12px ui-monospace,Menlo,monospace}'
            '.vn{font-weight:700;color:var(--ac)}'
            '.vd{margin-left:auto;font:700 11px ui-monospace,Menlo,monospace;'
            'padding:2px 8px;border-radius:10px;border:1px solid var(--bd)}'
            '.vok{color:var(--gn)}.vweak{color:var(--or)}.voff{color:var(--mu)}'
            '.bar{display:flex;align-items:center;gap:8px;margin:3px 0}'
            '.bn{width:150px;color:var(--mu);font-size:.9em;text-align:right}'
            '.bt{height:12px;border-radius:3px}'
            '.bv{font:12px ui-monospace,Menlo,monospace}'
            '.mg{margin-top:6px;color:var(--mu);font-size:.85em}'
            '.tk{display:inline-block;padding:2px 4px;margin:1px;border-radius:4px;'
            'font:12px/1.4 ui-monospace,Menlo,monospace;border:1px solid transparent}'
            '.ct{background:rgba(126,231,135,.14)}'
            '.sp{background:rgba(121,192,255,.22);color:var(--ac);border-color:var(--bd)}'
            '.ans{outline:2px solid var(--or);font-weight:700}'
            '.tk:hover{border-color:var(--ac)}'
            '.nav{position:fixed;right:14px;bottom:14px;display:flex;flex-direction:column;'
            'gap:6px;z-index:9998}'
            '.nav a{display:block;width:38px;height:38px;line-height:38px;text-align:center;'
            'background:var(--pan);border:1px solid var(--bd);border-radius:8px;color:var(--tx);'
            'text-decoration:none;font-size:16px;box-shadow:0 2px 8px rgba(0,0,0,.3)}'
            '.nav a:hover{border-color:var(--ac);color:var(--ac)}'
            '.diag-card{border:1px solid var(--bd);border-radius:8px;background:var(--pan);'
            'padding:10px 14px;margin:10px 0}'
            '.diag-card .dt{font-weight:700;color:var(--ac);margin-bottom:6px}'
            '.diag-card .dn{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;'
            'font:12px ui-monospace,Menlo,monospace;color:var(--tx)}'
            '.diag-card .dn b{color:var(--mu);font-weight:400}'
            '.diag-card .pill{display:inline-block;padding:2px 10px;border-radius:10px;'
            'font:700 11px ui-monospace,Menlo,monospace;border:1px solid var(--bd)}'
            '.pill-gn{color:var(--gn)}.pill-or{color:var(--or)}.pill-mu{color:var(--mu)}')

        nav = ('<div class="nav">'
               '<a href="#top" title="top">⤒</a>'
               '<a href="#strip" title="jump to token strip">↓</a>'
               '<a href="#top" title="up">↑</a>'
               '<a href="#bottom" title="bottom">⤓</a></div>')

        doc = (viz._HEAD.format(title=esc(title)).replace("</style>", css + "</style>") +
               f'<a id="top"></a>'
               f'<h1>{esc(title)}</h1>'
               f'<p class="sub">读的是<b>真实模板渲染的 token</b>（不是裸字符串）· '
               f'排名按 intended 义与最强竞争义的 margin · '
               f'目标义: <b>{esc(intended)}</b> · J-Lens @ layer {esc(self._layer(layer))}</p>'
               f'<h3 style="margin:1.2em 0 .3em;color:var(--ac)">模板变体排名（best first）</h3>'
               f'{rank_html}'
               f'{diag_section}'
               f'<a id="strip"></a>'
               f'<h3 style="margin:1.6em 0 .3em;color:var(--ac)">胜出变体的 token strip · [{esc(win)}]</h3>'
               f'<p class="sub"><span class="sp" style="padding:1px 4px">蓝色</span>=模板注入的特殊 token'
               f'（你看不见的那层）· '
               f'<span class="ans" style="padding:1px 4px">橙框</span>=答案位 · '
               f'hover 看该位置「欲言」的概念。</p>'
               f'<div style="line-height:2.2">{boxes}</div>'
               f'{nav}'
               f'<a id="bottom"></a></body></html>')
        if out_path:
            open(out_path, "w").write(doc)
        return doc
