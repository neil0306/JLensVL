"""Pure metric functions over per-item J-Lens race results.

A "per-item result" is `{layer: {concept_name: score}}` (the shape returned
by `JLensVL.concept_race` / `jlensvl.eval.runner.score_text_concepts`).
`target` is the winning concept's name; `distractors` is a list of rival
concept names. Every function here takes those three things directly and has
no dependency on `jlensvl.eval.stimuli` or on a live model -- they're plain,
independently testable functions over dicts of numbers.

Layer indices are assumed to sort ascending from shallow (early network) to
deep (late network) -- "earliest" below means smallest layer index.
"""
from __future__ import annotations

from typing import Optional


def _distractor_scores(layer_scores: dict, distractors) -> dict:
    return {d: layer_scores[d] for d in distractors if d in layer_scores}


def correct_at_layer(scores: dict, target: str, distractors) -> dict:
    """{layer: bool} -- True where `target`'s score beats every *scored*
    distractor at that layer. Layers where `target` itself wasn't scored are
    omitted entirely (there's nothing to judge). A layer where `target` was
    scored but none of `distractors` were is trivially True (nothing to
    lose to)."""
    out = {}
    for layer, layer_scores in scores.items():
        if target not in layer_scores:
            continue
        dscores = _distractor_scores(layer_scores, distractors)
        out[layer] = (layer_scores[target] > max(dscores.values())) if dscores else True
    return out


def first_correct_layer(scores: dict, target: str, distractors) -> Optional[int]:
    """Earliest layer where `correct_at_layer` is True, or None if the target
    never beats every scored distractor simultaneously."""
    correct = correct_at_layer(scores, target, distractors)
    winning = sorted(layer for layer, ok in correct.items() if ok)
    return winning[0] if winning else None


def margin_at_layer(scores: dict, target: str, distractors) -> dict:
    """{layer: target_score - best_distractor_score}. A layer is omitted if
    `target` wasn't scored there, or if none of `distractors` were (margin is
    undefined without a rival to measure against -- unlike `correct_at_layer`,
    which treats "no rival" as trivially correct)."""
    out = {}
    for layer, layer_scores in scores.items():
        if target not in layer_scores:
            continue
        dscores = _distractor_scores(layer_scores, distractors)
        if not dscores:
            continue
        out[layer] = layer_scores[target] - max(dscores.values())
    return out


def crossover_layer(scores: dict, target: str, distractors) -> Optional[int]:
    """Earliest layer where `target` overtakes its *leading rival*, mirroring
    `viz.race_chart_html`'s two-curve crossover (first layer where curve A
    rises above curve B).

    With >1 distractor there's no single fixed "curve B" a priori, so the
    rival is pinned down as the single distractor with the highest score at
    the *final* (largest-index) scored layer -- i.e. the one `target`
    ultimately has to beat -- and then we walk layers from the start looking
    for the first point where `target` is already ahead of *that* rival's
    curve.

    This can differ from `first_correct_layer`: that metric requires beating
    *every* distractor at once, layer by layer, so it can fire later than
    the target first passes the eventual leader (if some other distractor is
    still ahead at that point) or earlier (if the target is ahead of the
    eventual leader from the start but only clears every rival at some later
    layer).

    Returns None if there are no scored layers, the rival is never scored
    alongside the target at the final layer, or `target` never overtakes it.
    """
    layers = sorted(scores.keys())
    if not layers:
        return None
    final_scores = scores[layers[-1]]
    rival_candidates = [d for d in distractors if d in final_scores]
    if not rival_candidates:
        return None
    rival = max(rival_candidates, key=lambda d: final_scores[d])
    for layer in layers:
        layer_scores = scores[layer]
        if target in layer_scores and rival in layer_scores and layer_scores[target] > layer_scores[rival]:
            return layer
    return None


def aggregate(results, *, final_layer=None) -> dict:
    """Dataset-level rollup over `results`: a list of flat per-item dicts
    (see `jlensvl.eval.runner.EvalRunner.run`) each shaped like
    `{"target": str, "distractors": [str], "scores": {layer: {name: score}},
    "category": str, "skipped": bool, ...}`. Items with `skipped=True` or an
    empty `scores` dict are excluded from every statistic (but still counted
    in `n_items`).

    Returns a plain, JSON-serializable dict:

        {
          "n_items": int,
          "n_scored": int,                    # items with usable scores
          "accuracy_final_layer": float | None,
          "mean_first_correct_layer": float | None,
          "mean_peak_margin": float | None,
          "by_category": {category: {...same 5 keys...}},
        }

    `final_layer`, if given (e.g. the lens's last fitted layer), is the layer
    accuracy is measured at; for an item that has no score at exactly that
    layer, its own deepest scored layer is used instead so one missing layer
    doesn't drop the item from the accuracy statistic entirely.
    """

    def _rollup(items) -> dict:
        n_items = len(items)
        correct_flags = []
        first_layers = []
        peak_margins = []
        n_scored = 0
        for r in items:
            scores = r.get("scores") or {}
            if r.get("skipped") or not scores:
                continue
            n_scored += 1
            target, distractors = r["target"], r["distractors"]

            correct = correct_at_layer(scores, target, distractors)
            if correct:
                layer_for_acc = final_layer if final_layer in correct else max(correct.keys())
                correct_flags.append(bool(correct[layer_for_acc]))

            fcl = first_correct_layer(scores, target, distractors)
            if fcl is not None:
                first_layers.append(fcl)

            margins = margin_at_layer(scores, target, distractors)
            if margins:
                peak_margins.append(max(margins.values()))

        return {
            "n_items": n_items,
            "n_scored": n_scored,
            "accuracy_final_layer": (sum(correct_flags) / len(correct_flags)) if correct_flags else None,
            "mean_first_correct_layer": (sum(first_layers) / len(first_layers)) if first_layers else None,
            "mean_peak_margin": (sum(peak_margins) / len(peak_margins)) if peak_margins else None,
        }

    overall = _rollup(results)
    categories = sorted({r.get("category") or "uncategorized" for r in results})
    overall["by_category"] = {
        c: _rollup([r for r in results if (r.get("category") or "uncategorized") == c]) for c in categories
    }
    return overall
