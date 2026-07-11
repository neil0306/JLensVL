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
        return layer if layer is not None else self.jl.lens.source_layers[-2]

    def poised(self, prompt, *, layer=None, k=6):
        """What is the model poised to say at the end of `prompt`?
        Returns top-k tokens + a decisiveness margin (top1 - top2)."""
        jl = self.jl; jl._require_lens(); L = self._layer(layer)
        ll, _, _ = jl.lens.apply(jl.lm, prompt, positions=[-1], layers=[L], use_jacobian=True)
        vals, idx = ll[L][0].topk(k)
        toks = jl._decode(idx.tolist())
        return {"tokens": toks, "top1": toks[0], "margin": float(vals[0] - vals[1]), "layer": L}

    def sense_scores(self, prompt, senses, *, layer=None):
        """J-Lens score of each candidate sense at the answer position.
        `senses` = {name: [words]}. Higher = the model leans that way."""
        jl = self.jl; jl._require_lens(); L = self._layer(layer)
        ll, _, _ = jl.lens.apply(jl.lm, prompt, positions=[-1], layers=[L], use_jacobian=True)
        logits = ll[L][0]
        return {n: float(logits[jl._word_ids(w)].max()) for n, w in senses.items()}

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
    def _render(self, messages, *, add_generation_prompt=True, enable_thinking=None):
        """messages -> the exact chat-template-rendered string the model sees."""
        tok = self.jl.tok
        kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        if enable_thinking is not None:
            try:
                return tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kw)
            except TypeError:
                pass
        return tok.apply_chat_template(messages, **kw)

    def trace_rendered(self, messages, *, senses=None, enable_thinking=None,
                       add_generation_prompt=True, layer=None, topk=6):
        """Run the J-Lens on the REAL chat-template-rendered token sequence (not the
        raw string). Returns a per-token trace + segment/special tags + the sense
        scores at the answer position. This is the faithful thing to lens for a
        chat model — the model never sees your raw text, it sees the rendered tokens.
        """
        jl = self.jl; jl._require_lens()
        tok = jl.tok
        rendered = self._render(messages, add_generation_prompt=add_generation_prompt,
                                enable_thinking=enable_thinking)
        layers = jl.lens.source_layers
        ll, ml, ids = jl.lens.apply(jl.lm, rendered, positions=None, layers=layers, use_jacobian=True)
        if hasattr(ids, "dim") and ids.dim() > 1:
            ids = ids[0]
        ids = [int(i) for i in ids]
        toks = [tok.decode([i]) for i in ids]
        L = layer if layer is not None else layers[-2]
        special = set(getattr(tok, "all_special_ids", []) or [])
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
                        "special": is_sp, "role": role})
        ans = len(ids) - 1
        sc = None
        if senses:
            alog = ll[L][ans]
            sc = {n: float(alog[jl._word_ids(w)].max()) for n, w in senses.items()}
        return {"rendered": rendered, "tokens": toks, "per": per, "answer": ans,
                "layer": int(L), "senses": sc}

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
            '.nav a:hover{border-color:var(--ac);color:var(--ac)}')

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
