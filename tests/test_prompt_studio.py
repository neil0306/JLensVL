"""CPU-only tests for the PromptStudio facade.

Everything here runs with plain stubs — NO model, NO GPU, NO real index. We
inject fake `RetrievalLens` / `PromptHelper` objects into `PromptStudio` and
verify the three things the facade is responsible for:

  * scaffolding (pure-Python system+user variant generation from templates),
  * the chaining logic (suggest -> scaffold -> rank -> winner diagnostics),
  * self-contained HTML emission from a result dict.

The real `RetrievalLens` / `PromptHelper` are exercised by their own suites; here
we only care that PromptStudio calls them with the right shapes and assembles the
result correctly.
"""

from __future__ import annotations

import pytest

from jlensvl.prompt_studio import DEFAULT_STYLES, PromptStudio


# ── stubs ───────────────────────────────────────────────────────────────────
class FakeRetrieval:
    """Records the concept it was asked about and returns canned native words."""

    def __init__(self):
        self.calls = []
        self.rendered_calls = []

    def propose_concept(self, jl, concept, k=8, template="{}", **kw):
        self.calls.append((concept, k, template))
        return [{"word": f"{concept}-alt{i}", "score": 5.0 - i,
                 "source_tag": "vg", "source_sentence": "s", "layer": 1,
                 "count": 1} for i in range(k)]

    def propose_rendered(self, jl, messages, k=8, **kw):
        self.rendered_calls.append((messages, k))
        return [{"word": "rendered-alt", "score": 4.2}]


class FakeHelper:
    """A PromptHelper stand-in that scores variants by a fixed per-name margin
    and records every call so the chaining can be asserted."""

    def __init__(self, order=("role", "direct", "reasoned")):
        # earlier in `order` -> larger intended margin (so `order[0]` wins)
        self._order = list(order)
        self.compare_calls = []
        self.rank_calls = []
        self.sysreg_calls = []
        self.thinking_calls = []

    def _score(self, name, intended):
        # higher rank (lower index) => stronger intended sense
        i = self._order.index(name) if name in self._order else len(self._order)
        return {intended: 10.0 - 2.0 * i, "_competitor": 3.0}

    def compare_templates(self, base_messages, variants, senses, intended, *, layer=None):
        self.compare_calls.append((base_messages, variants, senses, intended, layer))
        rows = []
        for name in variants:
            sc = self._score(name, intended)
            comp = max(v for k, v in sc.items() if k != intended)
            rows.append({"name": name, "scores": sc, "intended": sc[intended],
                         "best_competitor": comp, "margin": sc[intended] - comp})
        rows.sort(key=lambda r: r["margin"], reverse=True)
        return rows

    def rank_prompts(self, variants, senses, intended, *, layer=None):
        self.rank_calls.append((variants, senses, intended, layer))
        items = variants.items() if isinstance(variants, dict) else [(p, p) for p in variants]
        rows = [{"name": n, "prompt": p, "scores": {intended: 7.0, "x": 2.0},
                 "margin": 5.0} for n, p in items]
        return rows

    def check_system_registers(self, messages, senses, intended, *, layer=None):
        self.sysreg_calls.append(messages)
        return {"with_system": 6.0, "without_system": 3.0, "delta": 3.0,
                "verdict": "registers"}

    def diagnose_thinking(self, messages, senses, intended, *, layer=None):
        self.thinking_calls.append(messages)
        return {"thinking_on": {"intended": 6.0, "margin": 3.0},
                "thinking_off": {"intended": 5.0, "margin": 1.0},
                "delta_margin": 2.0, "verdict": "helps"}


SENSES = {"high": ["severe", "critical"], "low": ["minor", "safe"]}


def _studio(order=("role", "direct", "reasoned"), with_retrieval=True):
    return PromptStudio(
        jl=object(),
        retrieval_lens=FakeRetrieval() if with_retrieval else None,
        prompt_helper=FakeHelper(order),
    )


# ── scaffolding (model-free) ─────────────────────────────────────────────────
def test_scaffold_default_three_variants():
    s = _studio()
    variants = s.scaffold_variants("Decide the hazard tier", ["hazard"])
    assert list(variants) == DEFAULT_STYLES[:3]
    for name, cfg in variants.items():
        msgs = cfg["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        # the task text is woven into both system and user
        assert "hazard tier" in cfg["system"].lower() or "hazard tier" in cfg["user"].lower()
        assert isinstance(cfg["enable_thinking"], bool)
        # every variant mentions the concept somewhere
        blob = (cfg["system"] + cfg["user"]).lower()
        assert "hazard" in blob


def test_scaffold_weaves_suggestions_as_hint():
    s = _studio()
    suggestions = {"hazard": [{"word": "danger"}, {"word": "risk"}, "peril"]}
    variants = s.scaffold_variants("Assess safety", ["hazard"],
                                   suggestions=suggestions)
    joined = " ".join(cfg["user"] + cfg["system"] for cfg in variants.values())
    assert "danger" in joined and "risk" in joined and "peril" in joined


def test_scaffold_respects_style_subset_and_rejects_unknown():
    s = _studio()
    variants = s.scaffold_variants("t", ["c"], styles=["direct"])
    assert list(variants) == ["direct"]
    with pytest.raises(ValueError, match="unknown style"):
        s.scaffold_variants("t", ["c"], styles=["nope"])


# ── suggest_words ────────────────────────────────────────────────────────────
def test_suggest_words_delegates_to_retrieval():
    s = _studio()
    words = s.suggest_words("hazard", k=4)
    assert len(words) == 4
    assert words[0]["word"] == "hazard-alt0"
    assert s.retrieval.calls == [("hazard", 4, "{}")]


def test_suggest_words_without_retrieval_returns_empty():
    s = _studio(with_retrieval=False)
    assert s.suggest_words("hazard") == []


def test_suggest_words_rendered_path():
    s = _studio()
    msgs = [{"role": "user", "content": "a photo with a hazard"}]
    out = s.suggest_words("hazard", k=3, messages=msgs)
    assert out and out[0]["word"] == "rendered-alt"
    assert s.retrieval.rendered_calls and s.retrieval.rendered_calls[0][0] == msgs


# ── rank ─────────────────────────────────────────────────────────────────────
def test_rank_rendered_uses_compare_templates_best_first():
    s = _studio(order=("role", "direct", "reasoned"))
    variants = s.scaffold_variants("t", ["c"])
    rows = s.rank(variants, SENSES, "high")
    assert [r["name"] for r in rows][0] == "role"       # order[0] wins
    assert rows == sorted(rows, key=lambda r: r["margin"], reverse=True)
    # it went through the chat-template (compare_templates) path
    assert s.helper.compare_calls and not s.helper.rank_calls


def test_rank_raw_string_path_uses_rank_prompts():
    s = _studio()
    rows = s.rank(["prompt a", "prompt b"], SENSES, "high", rendered=False)
    assert len(rows) == 2 and s.helper.rank_calls and not s.helper.compare_calls


def test_rank_rendered_requires_mapping():
    s = _studio()
    with pytest.raises(TypeError):
        s.rank(["just", "strings"], SENSES, "high", rendered=True)


def test_rank_empty_variants_raises():
    s = _studio()
    with pytest.raises(ValueError, match="no variants"):
        s.rank({}, SENSES, "high", rendered=True)


def test_rank_without_helper_errors():
    s = PromptStudio(jl=None, retrieval_lens=FakeRetrieval(), prompt_helper=None)
    with pytest.raises(RuntimeError, match="PromptHelper"):
        s.rank({"a": {"messages": []}}, SENSES, "high")


# ── run: the full chain ──────────────────────────────────────────────────────
def _spec(**over):
    spec = {"task": "Decide the hazard tier of a workplace photo",
            "concepts": ["hazard"], "senses": SENSES, "intended": "high", "k": 3}
    spec.update(over)
    return spec


def test_run_full_chain_assembles_result():
    s = _studio(order=("role", "direct", "reasoned"))
    res = s.run(_spec())
    # suggestions populated per concept
    assert res["suggestions"]["hazard"][0]["word"] == "hazard-alt0"
    # scaffolded variants + best-first ranking + winner
    assert set(res["variants"]) == set(DEFAULT_STYLES[:3])
    assert res["ranking"][0]["name"] == res["winner"] == "role"
    # winner diagnostics ran on the winner's messages
    assert res["diagnostics"]["system_registers"]["verdict"] == "registers"
    assert res["diagnostics"]["thinking"]["verdict"] == "helps"
    assert s.helper.sysreg_calls and s.helper.thinking_calls


def test_run_hint_flows_from_suggestions_into_variants():
    s = _studio()
    res = s.run(_spec())
    # the retrieved native word for the concept appears in the scaffolded prompts
    joined = " ".join(c["user"] + c["system"] for c in res["variants"].values())
    assert "hazard-alt0" in joined


def test_run_without_retrieval_skips_suggestions_but_still_ranks():
    s = _studio(with_retrieval=False)
    res = s.run(_spec())
    assert res["suggestions"] == {} or all(not v for v in res["suggestions"].values())
    assert res["winner"] is not None


def test_run_can_disable_suggest_and_diagnose():
    s = _studio()
    res = s.run(_spec(suggest=False, diagnose=False))
    assert res["suggestions"] == {}
    assert res["diagnostics"] == {}
    assert not s.helper.sysreg_calls and not s.helper.thinking_calls


def test_run_validates_spec():
    s = _studio()
    with pytest.raises(ValueError, match="task"):
        s.run({"senses": SENSES, "intended": "high"})
    with pytest.raises(ValueError, match="senses"):
        s.run({"task": "t"})
    with pytest.raises(ValueError, match="not in senses"):
        s.run({"task": "t", "senses": SENSES, "intended": "nope"})


def test_run_diagnostics_degrade_on_helper_error():
    s = _studio()

    def boom(*a, **k):
        raise RuntimeError("thinking unsupported")

    s.helper.diagnose_thinking = boom
    res = s.run(_spec())
    # the run completes; the failing diagnostic is captured, not fatal
    assert "error" in res["diagnostics"]["thinking"]
    assert res["winner"] is not None


# ── HTML emission ────────────────────────────────────────────────────────────
def test_to_html_is_self_contained_and_covers_sections(tmp_path):
    s = _studio()
    res = s.run(_spec())
    out = tmp_path / "report.html"
    doc = s.to_html(res, out_path=str(out))
    assert out.exists()
    written = out.read_text()
    assert written == doc
    # self-contained: no external resource references
    for bad in ("http://", "https://", "src=", "<link"):
        assert bad not in doc
    assert doc.strip().startswith("<!doctype html>") and doc.rstrip().endswith("</html>")
    # every stage is represented
    assert "hazard-alt0" in doc          # suggestions
    assert res["winner"] in doc          # ranking / winner name
    assert "margin" in doc               # ranking bars
    assert "registers" in doc            # system diagnostic
    assert "system" in doc and "user" in doc  # variant full text


def test_to_html_without_suggestions_or_diagnostics(tmp_path):
    s = _studio(with_retrieval=False)
    res = s.run(_spec(diagnose=False))
    doc = s.to_html(res)
    # still a valid, self-contained doc even with sections empty
    assert doc.strip().startswith("<!doctype html>")
    assert res["winner"] in doc
