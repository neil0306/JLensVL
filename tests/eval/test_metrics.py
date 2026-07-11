"""Unit tests for jlensvl.eval.metrics -- pure functions over synthetic
per-item score dicts. No model, no torch."""
from jlensvl.eval.metrics import (
    aggregate,
    correct_at_layer,
    crossover_layer,
    first_correct_layer,
    margin_at_layer,
)


# ---------------------------------------------------------------------------
# correct_at_layer / first_correct_layer / margin_at_layer
# ---------------------------------------------------------------------------

def test_correct_at_layer_basic():
    scores = {
        0: {"target": 1, "d1": 5, "d2": 0},
        1: {"target": 6, "d1": 5, "d2": 7},
        2: {"target": 8, "d1": 5, "d2": 6},
    }
    correct = correct_at_layer(scores, "target", ["d1", "d2"])
    assert correct == {0: False, 1: False, 2: True}


def test_correct_at_layer_no_distractors_scored_is_trivially_true():
    scores = {0: {"target": 1}}
    assert correct_at_layer(scores, "target", ["d1", "d2"]) == {0: True}


def test_correct_at_layer_omits_layers_missing_target():
    scores = {0: {"d1": 5}, 1: {"target": 1, "d1": 0}}
    correct = correct_at_layer(scores, "target", ["d1"])
    assert correct == {1: True}
    assert 0 not in correct


def test_first_correct_layer_matches_earliest_true():
    scores = {
        0: {"target": 1, "d1": 5, "d2": 0},
        1: {"target": 6, "d1": 5, "d2": 7},
        2: {"target": 8, "d1": 5, "d2": 6},
    }
    assert first_correct_layer(scores, "target", ["d1", "d2"]) == 2


def test_first_correct_layer_none_when_never_correct():
    scores = {0: {"target": 1, "d1": 5}, 1: {"target": 2, "d1": 9}}
    assert first_correct_layer(scores, "target", ["d1"]) is None


def test_first_correct_layer_none_when_target_never_scored():
    scores = {0: {"d1": 5}, 1: {"d1": 9}}
    assert first_correct_layer(scores, "target", ["d1"]) is None


def test_margin_at_layer_values_and_omission():
    scores = {
        0: {"target": 1, "d1": 5, "d2": 0},
        1: {"target": 10, "d1": 2, "d2": 3},
        2: {"target": 4},  # no distractor scored -> omitted (undefined margin)
    }
    margins = margin_at_layer(scores, "target", ["d1", "d2"])
    assert margins == {0: 1 - 5, 1: 10 - 3}
    assert 2 not in margins


# ---------------------------------------------------------------------------
# crossover_layer -- and its deliberate divergence from first_correct_layer
# ---------------------------------------------------------------------------

def test_crossover_layer_can_precede_first_correct_layer():
    """target is ahead of the *eventual* leading rival (d2) from layer 0, but
    only clears every rival (d1 included) at layer 2 -- crossover fires
    earlier than first_correct because it only tracks the final-layer
    leader, not "beats everyone simultaneously"."""
    scores = {
        0: {"target": 1, "d1": 5, "d2": 0},
        1: {"target": 6, "d1": 5, "d2": 7},
        2: {"target": 8, "d1": 5, "d2": 6},  # final layer: leading rival = d2 (6)
    }
    assert crossover_layer(scores, "target", ["d1", "d2"]) == 0
    assert first_correct_layer(scores, "target", ["d1", "d2"]) == 2


def test_crossover_layer_requires_rival_present_at_layer():
    scores = {
        0: {"target": 1, "d1": 2},  # rival (d2) not scored at layer 0 -> skip
        1: {"target": 10, "d1": 2, "d2": 3},
    }
    assert crossover_layer(scores, "target", ["d1", "d2"]) == 1


def test_crossover_layer_none_when_no_layers():
    assert crossover_layer({}, "target", ["d1"]) is None


def test_crossover_layer_none_when_rival_never_at_final_layer():
    scores = {0: {"target": 5}}  # no distractor ever scored
    assert crossover_layer(scores, "target", ["d1"]) is None


def test_crossover_layer_none_when_target_never_overtakes():
    scores = {0: {"target": 1, "d1": 5}, 1: {"target": 2, "d1": 9}}
    assert crossover_layer(scores, "target", ["d1"]) is None


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def _item(id_, category, target, distractors, scores, skipped=False):
    return {
        "id": id_,
        "category": category,
        "target": target,
        "distractors": distractors,
        "scores": scores,
        "skipped": skipped,
        "reason": None,
    }


def test_aggregate_basic_accuracy_and_means():
    results = [
        _item("a", "cat1", "t", ["d"], {0: {"t": 1, "d": 5}, 1: {"t": 9, "d": 2}}),
        _item("b", "cat1", "t", ["d"], {0: {"t": 9, "d": 1}, 1: {"t": 9, "d": 1}}),
    ]
    out = aggregate(results, final_layer=1)
    assert out["n_items"] == 2
    assert out["n_scored"] == 2
    # item a: correct at layer1 only; item b: correct both layers -> both correct @ final_layer=1
    assert out["accuracy_final_layer"] == 1.0
    # first_correct: a->1, b->0 ; mean = 0.5
    assert out["mean_first_correct_layer"] == 0.5
    # peak margin: a-> max(1-5=-4, 9-2=7)=7 ; b-> max(8,8)=8 ; mean=7.5
    assert out["mean_peak_margin"] == 7.5


def test_aggregate_skips_skipped_and_empty_items():
    results = [
        _item("a", "cat1", "t", ["d"], {}, skipped=True),
        _item("b", "cat1", "t", ["d"], {}),  # empty scores, not flagged skipped
        _item("c", "cat1", "t", ["d"], {0: {"t": 5, "d": 1}}),
    ]
    out = aggregate(results)
    assert out["n_items"] == 3
    assert out["n_scored"] == 1
    assert out["accuracy_final_layer"] == 1.0


def test_aggregate_final_layer_fallback_to_item_max_layer():
    """final_layer=5 doesn't exist for this item; it should fall back to the
    item's own deepest scored layer instead of dropping the item."""
    results = [_item("a", "cat1", "t", ["d"], {0: {"t": 1, "d": 5}, 2: {"t": 9, "d": 1}})]
    out = aggregate(results, final_layer=5)
    assert out["accuracy_final_layer"] == 1.0  # layer 2 (max available) is correct


def test_aggregate_by_category_breakdown():
    results = [
        _item("a", "cap", "t", ["d"], {0: {"t": 9, "d": 1}}),
        _item("b", "cap", "t", ["d"], {0: {"t": 1, "d": 9}}),
        _item("c", "color", "t", ["d"], {0: {"t": 9, "d": 1}}),
    ]
    out = aggregate(results)
    assert set(out["by_category"]) == {"cap", "color"}
    assert out["by_category"]["cap"]["n_items"] == 2
    assert out["by_category"]["cap"]["accuracy_final_layer"] == 0.5
    assert out["by_category"]["color"]["n_items"] == 1
    assert out["by_category"]["color"]["accuracy_final_layer"] == 1.0


def test_aggregate_empty_results():
    out = aggregate([])
    assert out["n_items"] == 0
    assert out["n_scored"] == 0
    assert out["accuracy_final_layer"] is None
    assert out["mean_first_correct_layer"] is None
    assert out["mean_peak_margin"] is None
    assert out["by_category"] == {}


def test_aggregate_uses_uncategorized_bucket_for_missing_category():
    results = [_item("a", "", "t", ["d"], {0: {"t": 9, "d": 1}})]
    out = aggregate(results)
    assert "uncategorized" in out["by_category"]
