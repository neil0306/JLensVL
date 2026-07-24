"""PromptStudio — a one-stop "super prompt helper" facade.

This chains the three existing JLensVL pieces into a single usable workflow.
Given a TASK and a target concept (plus optional style hints), it:

1. **PROPOSES** the model's own native words/alternatives for the concept via
   `RetrievalLens` (``suggest_words`` -> ``propose_concept`` / ``propose_rendered``).
2. **SCAFFOLDS** 2-3 candidate ``system`` + ``user`` prompt variants from a small
   set of style templates (``scaffold_variants`` — pure Python, no model).
3. **RANKS / DIAGNOSES** the variants with `PromptHelper` on the *real*
   chat-template path (``compare_templates`` for the ranking, plus
   ``check_system_registers`` / ``diagnose_thinking`` on the winner).
4. Emits a **self-contained HTML report** (``to_html``) — no external deps,
   dark/light, offline — mirroring the rest of JLensVL's look.

The heavy pieces (`RetrievalLens`, `PromptHelper`, `JLensVL`) are only *used*
here, never reimplemented. Every one of them is injectable, so the chaining
logic, the scaffolding and the HTML emission are all testable on CPU with plain
stubs — no model, no GPU, no index.
"""

from __future__ import annotations

import html
import math
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence, Union


# ── style scaffolds (pure text; the SCAFFOLD step is model-free) ────────────
#
# Each style is a system/user template pair + a default thinking mode. They are
# rendered with ``.format(task=…, concepts=…, hint=…)`` — a deliberately small,
# readable set so a human can see (and edit) exactly what is being A/B'd.
_STYLES: "OrderedDict[str, dict]" = OrderedDict((
    ("direct", {
        "system": "You are a precise, concise assistant. {task}",
        "user": "{task}\n\nConcept in focus: {concepts}.{hint}\n"
                "Give a direct, decisive answer.",
        "enable_thinking": False,
    }),
    ("role", {
        "system": "You are a seasoned domain expert on {concepts}. {task} "
                  "Answer with the confidence of a specialist.",
        "user": "{task}{hint}",
        "enable_thinking": False,
    }),
    ("reasoned", {
        "system": "You are a careful analyst. {task} "
                  "Reason step by step, then commit to a single answer.",
        "user": "{task}\n\nFocus concept(s): {concepts}.{hint}\n"
                "Think it through, then answer.",
        "enable_thinking": True,
    }),
))

#: default ordered style names (first `max_variants` are used unless overridden)
DEFAULT_STYLES = list(_STYLES)


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _word_text(w) -> str:
    """A suggestion may be a plain string or a `propose_concept` dict — get the
    surface word either way."""
    if isinstance(w, Mapping):
        return str(w.get("word", "")).strip()
    return str(w).strip()


class PromptStudio:
    """Facade chaining `RetrievalLens` (word proposer) + `PromptHelper`
    (J-Lens ranking/diagnostics) + `JLensVL` (the fitted model) into a single
    "given a task + concept, suggest words, scaffold prompts, rank them, report"
    workflow.

    Args:
        jl: a fitted `JLensVL` (or a stub exposing ``.tok`` / the bits
            `PromptHelper` needs). Optional if you inject both helpers.
        retrieval_lens: a `RetrievalLens` (word proposer). Optional — without it
            ``suggest_words`` returns ``[]`` and scaffolding proceeds without
            model-native hints.
        prompt_helper: a `PromptHelper`. Defaults to ``PromptHelper(jl)`` when
            ``jl`` is given.
    """

    def __init__(self, jl=None, *, retrieval_lens=None, prompt_helper=None):
        self.jl = jl
        self.retrieval = retrieval_lens
        if prompt_helper is None and jl is not None:
            from .prompt_helper import PromptHelper  # lazy: keep import light
            prompt_helper = PromptHelper(jl)
        self.helper = prompt_helper

    # ---------- construction convenience ----------
    @classmethod
    def from_jl(cls, jl, index_path: Optional[str] = None) -> "PromptStudio":
        """Build a studio around a live `JLensVL`, optionally loading a saved
        `RetrievalIndex` from ``index_path`` for the word-proposer step."""
        retrieval = None
        if index_path:
            from .retrieval_lens import RetrievalLens  # lazy
            retrieval = RetrievalLens.load(index_path)
        return cls(jl, retrieval_lens=retrieval)

    # ---------- 1. propose model-native words ----------
    def suggest_words(self, concept: str, k: int = 8, *, template: str = "{}",
                      messages=None, **kw) -> list[dict]:
        """Model-native words/alternatives the model associates with ``concept``.

        Delegates to `RetrievalLens.propose_concept` (or `propose_rendered` when
        a chat ``messages`` list is given). Returns the proposer's ranked list of
        ``{word, score, ...}`` dicts, or ``[]`` if no `RetrievalLens` is wired in.
        """
        if self.retrieval is None:
            return []
        if messages is not None:
            return list(self.retrieval.propose_rendered(
                self.jl, messages, k=k, **kw))
        return list(self.retrieval.propose_concept(
            self.jl, concept, k=k, template=template, **kw))

    # ---------- 2. scaffold candidate prompts (model-free) ----------
    def scaffold_variants(self, task: str, concepts,
                          styles: Optional[Sequence[str]] = None, *,
                          suggestions: Optional[Mapping[str, Sequence]] = None,
                          max_variants: int = 3,
                          hint_k: int = 4) -> "OrderedDict[str, dict]":
        """Build 2-3 candidate ``system`` + ``user`` prompt variants from the
        style templates. Pure Python — no model, no forward pass.

        Args:
            task: the task description woven into every variant.
            concepts: concept name(s) the prompt should steer toward.
            styles: which style names to instantiate (subset of
                ``DEFAULT_STYLES``); defaults to the first ``max_variants``.
            suggestions: optional ``{concept: [word|dict, ...]}`` of model-native
                words (from `suggest_words`) — the top ``hint_k`` are woven into
                each variant as a "consider related notions" hint.
            max_variants: cap when ``styles`` is not given.
            hint_k: how many suggested words to include in the hint.

        Returns an ordered ``{style_name: {"messages", "enable_thinking",
        "system", "user"}}`` dict — directly consumable by `rank` /
        `PromptHelper.compare_templates`.
        """
        concepts = _as_list(concepts)
        concept_str = ", ".join(concepts) if concepts else task
        if styles is None:
            styles = DEFAULT_STYLES[:max_variants]
        unknown = [s for s in styles if s not in _STYLES]
        if unknown:
            raise ValueError(
                f"unknown style(s) {unknown}; known styles: {DEFAULT_STYLES}")

        # Build a single "consider related notions: a, b, c" hint from the
        # highest-ranked suggested words across all concepts (dedup, order-kept).
        hint = ""
        if suggestions:
            seen, words = set(), []
            for c in concepts or list(suggestions):
                for w in _as_list(suggestions.get(c)):
                    t = _word_text(w)
                    tl = t.lower()
                    if t and tl not in seen:
                        seen.add(tl)
                        words.append(t)
            words = words[:hint_k]
            if words:
                hint = " Consider related notions: " + ", ".join(words) + "."

        out: "OrderedDict[str, dict]" = OrderedDict()
        for name in styles:
            tpl = _STYLES[name]
            fmt = dict(task=task.strip(), concepts=concept_str, hint=hint)
            system = tpl["system"].format(**fmt).strip()
            user = tpl["user"].format(**fmt).strip()
            out[name] = {
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "enable_thinking": tpl["enable_thinking"],
                "system": system,
                "user": user,
            }
        return out

    # ---------- 3. rank / diagnose ----------
    def rank(self, variants, senses, intended, *, layer=None,
             rendered: bool = True, base_messages=None) -> list[dict]:
        """Rank ``variants`` by how strongly each steers the model to the
        ``intended`` sense versus its strongest competitor.

        With ``rendered=True`` (default, the faithful chat-template path) this
        uses `PromptHelper.compare_templates` over the real rendered token
        sequence; ``variants`` must be the ``{name: {"messages", ...}}`` mapping
        `scaffold_variants` returns. With ``rendered=False`` it falls back to
        `PromptHelper.rank_prompts` over raw prompt strings (``variants`` a list
        or ``{name: prompt}``). Returns best-first rows.
        """
        if self.helper is None:
            raise RuntimeError("no PromptHelper: pass jl= or prompt_helper= to "
                               "PromptStudio()")
        if not rendered:
            return self.helper.rank_prompts(variants, senses, intended,
                                            layer=layer)
        if not isinstance(variants, Mapping):
            raise TypeError("rendered ranking needs a {name: {'messages':…}} "
                            "mapping (from scaffold_variants); got "
                            f"{type(variants).__name__}")
        if not variants:
            raise ValueError("no variants to rank (empty mapping — check "
                             "`styles`)")
        if base_messages is None:
            first = next(iter(variants.values()))
            base_messages = first.get("messages") if isinstance(first, Mapping) \
                else first
        return self.helper.compare_templates(base_messages, variants, senses,
                                             intended, layer=layer)

    # ---------- counterfactual / tool surfaces (thin passthroughs) ----------
    def poised_continuation(self, messages, prefill, *, reasoning=False,
                            layer=None, senses=None, k=6):
        """Surface `PromptHelper.poised_continuation`: if the assistant had
        already committed to `prefill`, what is it poised to say next?"""
        if self.helper is None:
            raise RuntimeError("no PromptHelper: pass jl= or prompt_helper=")
        return self.helper.poised_continuation(
            messages, prefill, reasoning=reasoning, layer=layer, senses=senses, k=k)

    def poised_tool(self, messages, tools, *, layer=None, tool_names=None,
                    arg_names=None, enable_thinking=None):
        """Surface `PromptHelper.poised_tool`: which tool / arg is the model
        poised toward for a tool-calling prompt?"""
        if self.helper is None:
            raise RuntimeError("no PromptHelper: pass jl= or prompt_helper=")
        return self.helper.poised_tool(
            messages, tools, layer=layer, tool_names=tool_names,
            arg_names=arg_names, enable_thinking=enable_thinking)

    def _diagnose_winner(self, messages, senses, intended, layer) -> dict:
        """Winner-only diagnostics: does the system message land, does thinking
        help. Each is guarded so a stub/path that lacks the feature degrades to
        ``None`` rather than aborting the whole run."""
        diag: dict[str, Any] = {}
        if self.helper is None:
            return diag
        if any(m.get("role") == "system" for m in messages):
            try:
                diag["system_registers"] = self.helper.check_system_registers(
                    messages, senses, intended, layer=layer)
            except Exception as exc:  # noqa: BLE001 - degrade, don't abort
                diag["system_registers"] = {"error": str(exc)}
        try:
            diag["thinking"] = self.helper.diagnose_thinking(
                messages, senses, intended, layer=layer)
        except Exception as exc:  # noqa: BLE001
            diag["thinking"] = {"error": str(exc)}
        return diag

    # ---------- the one-stop chain ----------
    def run(self, task_spec: Mapping[str, Any]) -> dict:
        """Run the whole chain from a small task spec and return a result dict.

        ``task_spec`` keys:
            task (str, required): the task description.
            concepts | concept: concept name(s) to propose words for / steer to.
            senses (dict, required): ``{sense_name: [words]}`` candidate meanings.
            intended (str, required): which sense the prompt should elicit.
            styles (list, optional): style names to scaffold.
            k (int): number of native words per concept (default 8).
            template (str): template for `propose_concept` (default ``"{}"``).
            layer (int|None): J-Lens layer for ranking/diagnostics.
            diagnose (bool): run winner diagnostics (default True).
            suggest (bool): run the word-proposer step (default True).

        Returns a dict with ``task, concepts, senses, intended, suggestions,
        variants, ranking, winner, diagnostics, layer``.
        """
        task = task_spec.get("task")
        if not task:
            raise ValueError("task_spec needs a non-empty 'task'")
        concepts = _as_list(task_spec.get("concepts", task_spec.get("concept")))
        senses = task_spec.get("senses")
        intended = task_spec.get("intended")
        if not senses or intended is None:
            raise ValueError("task_spec needs 'senses' and 'intended'")
        if intended not in senses:
            raise ValueError(
                f"intended sense {intended!r} not in senses {list(senses)}")
        layer = task_spec.get("layer")
        k = int(task_spec.get("k", 8))
        template = task_spec.get("template", "{}")
        styles = task_spec.get("styles")

        # 1. propose model-native words per concept
        suggestions: "OrderedDict[str, list]" = OrderedDict()
        if task_spec.get("suggest", True) and self.retrieval is not None:
            for c in concepts:
                try:
                    suggestions[c] = self.suggest_words(c, k=k, template=template)
                except Exception as exc:  # noqa: BLE001 - proposer is best-effort
                    suggestions[c] = []
                    suggestions.setdefault("_errors", {})[c] = str(exc)  # type: ignore[index]

        # 2. scaffold candidate prompts (model-free)
        variants = self.scaffold_variants(task, concepts, styles,
                                          suggestions=suggestions or None)

        # 3. rank on the real chat-template path
        ranking = self.rank(variants, senses, intended, layer=layer,
                            rendered=True)
        winner = ranking[0]["name"] if ranking else None

        # 4. winner diagnostics
        diagnostics: dict = {}
        if winner is not None and task_spec.get("diagnose", True):
            diagnostics = self._diagnose_winner(
                variants[winner]["messages"], senses, intended, layer)

        return {
            "task": task,
            "concepts": concepts,
            "senses": senses,
            "intended": intended,
            "layer": layer,
            "suggestions": suggestions,
            "variants": variants,
            "ranking": ranking,
            "winner": winner,
            "diagnostics": diagnostics,
        }

    # ---------- 4. self-contained HTML report ----------
    def to_html(self, result: Mapping[str, Any], out_path: Optional[str] = None,
                *, title: str = "JLensVL PromptStudio report") -> str:
        """Render a `run` result dict to a single self-contained HTML report
        (inline CSS, no external deps, dark/light, offline). Writes to
        ``out_path`` if given; always returns the HTML string.

        This works purely off the result dict — no model call — so it is testable
        with stub-produced results.
        """
        from . import viz  # lazy: mirror PromptHelper.report_html

        def esc(s):
            return html.escape(str(s))

        intended = result.get("intended", "")
        layer = result.get("layer")
        layer_txt = "auto" if layer is None else str(layer)

        # --- suggestions section ---
        sug_html = ""
        for concept, words in (result.get("suggestions") or {}).items():
            if concept == "_errors":
                continue
            chips = ""
            for w in words:
                wt = esc(_word_text(w))
                sc = w.get("score") if isinstance(w, Mapping) else None
                sct = f' <span class="sc">{sc:.2f}</span>' if isinstance(sc, (int, float)) else ""
                chips += f'<span class="chip">{wt}{sct}</span>'
            if not chips:
                chips = '<span class="chip muted">（无建议 / no proposer）</span>'
            sug_html += (f'<div class="sug"><div class="sc-h">概念 '
                         f'<b>{esc(concept)}</b> 的模型原生词</div>{chips}</div>')

        # --- ranking bars (mirror report_html's grouped bars) ---
        rows = list(result.get("ranking") or [])
        allv = [v for r in rows for v in (r.get("scores") or {}).values()
                if v is not None and math.isfinite(v)]
        vmax = max(allv) if allv else 1e-6
        vmax = vmax if vmax > 0 else 1e-6

        def verdict(m):
            return "✓ CLEAR" if m >= 3 else ("~ weak" if m > 0 else "✗ OFF-TARGET")

        rank_html = ""
        for i, r in enumerate(rows, 1):
            m = r.get("margin")
            m = float("-inf") if m is None else m
            vd = verdict(m)
            vcls = "vok" if m >= 3 else ("vweak" if m > 0 else "voff")
            bars = ""
            scores = r.get("scores") or {}
            for s, v in sorted(scores.items(),
                               key=lambda x: -(x[1] if x[1] is not None else -1e9)):
                val = v if (v is not None and math.isfinite(v)) else 0.0
                w = max(2, int(round(max(val, 0.0) / vmax * 240)))
                hl = "background:var(--gn)" if s == intended else "background:var(--mu)"
                tag = " ← intended" if s == intended else ""
                bars += (f'<div class="bar"><span class="bn">{esc(s)}{tag}</span>'
                         f'<span class="bt" style="width:{w}px;{hl}"></span>'
                         f'<span class="bv">{val:.2f}</span></div>')
            mtxt = f"{m:+.2f}" if m != float("-inf") else "n/a"
            rank_html += (f'<div class="var"><div class="vh">'
                          f'<span class="rk">#{i}</span>'
                          f'<span class="vn">{esc(r.get("name"))}</span>'
                          f'<span class="vd {vcls}">{esc(vd)}</span></div>'
                          f'{bars}'
                          f'<div class="mg">margin (intended − best competitor) '
                          f'= {esc(mtxt)}</div></div>')

        # --- diagnostics cards ---
        diag = result.get("diagnostics") or {}
        diag_html = ""
        sr = diag.get("system_registers")
        if isinstance(sr, Mapping) and "error" not in sr:
            pill = "pill-gn" if sr.get("verdict") == "registers" else "pill-or"
            diag_html += (
                '<div class="diag-card"><div class="dt">System 消息是否「落地」？</div>'
                '<div class="dn">'
                f'<span><b>with_system:</b> {sr.get("with_system", float("nan")):.2f}</span>'
                f'<span><b>without_system:</b> {sr.get("without_system", float("nan")):.2f}</span>'
                f'<span><b>delta:</b> {sr.get("delta", float("nan")):+.2f}</span>'
                f'<span class="pill {pill}">{esc(sr.get("verdict"))}</span>'
                '</div></div>')
        dt = diag.get("thinking")
        if isinstance(dt, Mapping) and "error" not in dt:
            dv = dt.get("verdict")
            pill = ("pill-gn" if dv == "helps"
                    else "pill-or" if dv == "hurts" else "pill-mu")
            on = dt.get("thinking_on", {}) or {}
            off = dt.get("thinking_off", {}) or {}
            diag_html += (
                '<div class="diag-card"><div class="dt">Thinking 模式是否有帮助？</div>'
                '<div class="dn">'
                f'<span><b>on margin:</b> {on.get("margin", float("nan")):+.2f}</span>'
                f'<span><b>off margin:</b> {off.get("margin", float("nan")):+.2f}</span>'
                f'<span><b>delta_margin:</b> {dt.get("delta_margin", float("nan")):+.2f}</span>'
                f'<span class="pill {pill}">{esc(dv)}</span>'
                '</div></div>')

        # --- winning-variant prompt text ---
        winner = result.get("winner")
        variants = result.get("variants") or {}
        prompt_html = ""
        for name, cfg in variants.items():
            if not isinstance(cfg, Mapping):
                continue
            is_win = " win" if name == winner else ""
            th = cfg.get("enable_thinking")
            th_txt = "" if th is None else f' · thinking={th}'
            prompt_html += (
                f'<div class="pv{is_win}"><div class="pvh">{esc(name)}'
                f'{"  ← winner" if name == winner else ""}'
                f'<span class="pvm">{esc(th_txt)}</span></div>'
                f'<div class="pp"><span class="pl">system</span>'
                f'<code>{esc(cfg.get("system", ""))}</code></div>'
                f'<div class="pp"><span class="pl">user</span>'
                f'<code>{esc(cfg.get("user", ""))}</code></div></div>')

        css = (
            '.sug{border:1px solid var(--bd);border-radius:8px;background:var(--pan);'
            'padding:10px 12px;margin:8px 0}'
            '.sc-h{color:var(--mu);font-size:.9em;margin-bottom:6px}'
            '.chip{display:inline-block;padding:2px 9px;margin:2px;border-radius:12px;'
            'border:1px solid var(--bd);background:rgba(126,231,135,.12);'
            'font:12px ui-monospace,Menlo,monospace}'
            '.chip.muted{color:var(--mu);background:transparent}'
            '.chip .sc{color:var(--mu)}'
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
            '.diag-card{border:1px solid var(--bd);border-radius:8px;background:var(--pan);'
            'padding:10px 14px;margin:10px 0}'
            '.diag-card .dt{font-weight:700;color:var(--ac);margin-bottom:6px}'
            '.diag-card .dn{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;'
            'font:12px ui-monospace,Menlo,monospace;color:var(--tx)}'
            '.diag-card .dn b{color:var(--mu);font-weight:400}'
            '.diag-card .pill{display:inline-block;padding:2px 10px;border-radius:10px;'
            'font:700 11px ui-monospace,Menlo,monospace;border:1px solid var(--bd)}'
            '.pill-gn{color:var(--gn)}.pill-or{color:var(--or)}.pill-mu{color:var(--mu)}'
            '.pv{border:1px solid var(--bd);border-radius:8px;background:var(--pan);'
            'padding:8px 12px;margin:8px 0}'
            '.pv.win{border-color:var(--gn)}'
            '.pvh{font-weight:700;color:var(--ac);margin-bottom:4px}'
            '.pvm{color:var(--mu);font-weight:400;font-size:.85em}'
            '.pp{display:flex;gap:8px;margin:3px 0;align-items:baseline}'
            '.pl{width:52px;color:var(--mu);font-size:.8em;flex:none;text-align:right}'
            '.pp code{white-space:pre-wrap;font:12px ui-monospace,Menlo,monospace}')

        def section(h):
            return f'<h3 style="margin:1.3em 0 .3em;color:var(--ac)">{h}</h3>'

        doc = (viz._HEAD.format(title=esc(title)).replace("</style>", css + "</style>") +
               f'<h1>{esc(title)}</h1>'
               f'<p class="sub">任务: <b>{esc(result.get("task"))}</b> · '
               f'目标义: <b>{esc(intended)}</b> · 概念: {esc(", ".join(result.get("concepts") or []))} · '
               f'J-Lens @ layer {esc(layer_txt)}</p>'
               + (section("① 模型原生词建议（retrieval lens）") + sug_html if sug_html else "")
               + section("② 候选提示词变体排名（best first）")
               + (rank_html or '<p class="sub">（无排名 / no ranking）</p>')
               + (section(f"③ 诊断卡片 · [{esc(winner)}]") + diag_html if diag_html else "")
               + section("④ 变体全文（system + user）")
               + (prompt_html or '<p class="sub">（无变体）</p>')
               + '</body></html>')
        if out_path:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(doc)
        return doc
