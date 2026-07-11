"""jlensvl.viz: pure data -> self-contained HTML string functions.

`race_chart_html` and `rendered_strip_html` take plain dicts/lists (no model,
no lens) so they're fully unit-testable offline. `slice_grid_html`,
`slice_grid_image_html` and `decision_gif` all require a live model/lens
forward pass and are out of scope here (covered later by integration tests).
"""
from __future__ import annotations

from jlensvl import viz


def _assert_self_contained_html(doc: str):
    assert isinstance(doc, str)
    assert len(doc) > 0
    lowered = doc.lower()
    assert "<!doctype" in lowered or "<html" in lowered
    assert "<style" in lowered
    # self-contained: no external network resource references
    assert "http://" not in doc
    assert "https://" not in doc


def test_race_chart_html_is_self_contained_and_embeds_concepts():
    race = {
        0: {"cat": 1.0, "dog": 4.0},
        1: {"cat": 2.0, "dog": 3.5},
        2: {"cat": 3.5, "dog": 3.0},
        3: {"cat": 5.0, "dog": 2.0},
    }
    doc = viz.race_chart_html(race, "cat", "dog", title="cat vs dog race")

    _assert_self_contained_html(doc)
    assert "<svg" in doc
    assert "cat" in doc
    assert "dog" in doc
    assert "cat vs dog race" in doc


def test_race_chart_html_computes_default_crossover():
    # concept_b leads until layer 3, where concept_a overtakes -> crossover=3.
    race = {
        0: {"a": 1.0, "b": 5.0},
        1: {"a": 2.0, "b": 4.0},
        2: {"a": 3.0, "b": 3.5},
        3: {"a": 6.0, "b": 3.0},
    }
    doc = viz.race_chart_html(race, "a", "b")
    _assert_self_contained_html(doc)
    assert "L3" in doc  # crossover-layer marker rendered


def test_race_chart_html_writes_out_path(tmp_path):
    race = {0: {"a": 1.0, "b": 2.0}, 1: {"a": 3.0, "b": 1.0}}
    out = tmp_path / "race.html"
    doc = viz.race_chart_html(race, "a", "b", out_path=str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == doc


def _rendered_strip_trace():
    return {
        "layer": 12,
        "answer": 2,
        "per": [
            {"tok": "Hello", "top": ["Hello", "Hi"], "special": False},
            {"tok": "<image>", "top": ["img"], "special": True, "role": "user"},
            {"tok": "world", "top": ["world", "earth"], "special": False},
        ],
        "senses": {"cat": 5.0, "dog": 3.0},
    }


def test_rendered_strip_html_is_self_contained_and_embeds_tokens():
    trace = _rendered_strip_trace()
    doc = viz.rendered_strip_html(trace, intended="cat", title="strip test")

    _assert_self_contained_html(doc)
    assert "Hello" in doc
    assert "world" in doc
    assert "strip test" in doc


def test_rendered_strip_html_embeds_sense_scores_and_marks_intended():
    trace = _rendered_strip_trace()
    doc = viz.rendered_strip_html(trace, intended="cat")

    _assert_self_contained_html(doc)
    assert "cat" in doc
    assert "dog" in doc
    assert "&#8592; intended" in doc or "← intended" in doc or "intended" in doc


def test_rendered_strip_html_without_senses_still_renders():
    trace = _rendered_strip_trace()
    del trace["senses"]
    doc = viz.rendered_strip_html(trace)
    _assert_self_contained_html(doc)
    assert "Hello" in doc
