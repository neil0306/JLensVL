"""Drives a fitted `JLensVL` over a `StimulusSet` and turns raw per-layer
concept scores into metrics.

Two scoring paths, chosen per item by whether `Stimulus.image` is set:
  * image items go through `JLensVL.concept_race` (unmodified, as shipped).
  * text-only items go through `score_text_concepts` below, a small helper
    that mirrors `concept_race`'s scoring for a *text* prompt: one forward
    pass via `JacobianLens.apply` (see `jlensvl/core.py:trace` and
    `jacobian-lens/jlens/lens.py:JacobianLens.apply`, which already returns
    per-layer `[vocab]` logits for a text prompt in one call), then the same
    "max logit over the concept's word ids" scoring `concept_race` uses. This
    reuses the existing transport+unembed math instead of reimplementing it.

`EvalRunner` degrades gracefully rather than raising:
  * an item is *skipped* (`skipped=True`, `scores={}`, a `reason`) when its
    *target* concept has no single-token surface form under the tokenizer --
    there is nothing to measure.
  * a *distractor* concept with no single-token surface form is silently
    dropped from that item's distractor list (recorded in `reason`) rather
    than skipping the whole item -- losing one rival weakens the signal, it
    doesn't invalidate it. `JLensVL.concept_race` itself has no such guard
    (an empty word-id list would crash its `logits[i].max()`), so this
    filtering happens here, before either scoring path is called.
"""
from __future__ import annotations

from .metrics import aggregate
from .stimuli import Stimulus, StimulusSet


def score_text_concepts(jl, prompt, concepts, *, layers=None, position=-1):
    """Text-only analogue of `JLensVL.concept_race`.

    `concepts` = {name: [words]}. Runs one forward pass over `prompt` through
    the fitted lens and returns `{layer: {name: score}}`, where `score` is
    the max logit among the concept's single-token word ids (same scoring
    `concept_race` does at each layer). Concepts with zero single-token word
    ids are dropped from the output rather than raising. Returns `{}` if no
    concept in `concepts` has any scorable word.
    """
    jl._require_lens()
    layers = list(layers) if layers is not None else jl.lens.source_layers
    cid = {n: jl._word_ids(w) for n, w in concepts.items()}
    cid = {n: ids for n, ids in cid.items() if ids}
    if not cid:
        return {}
    lens_logits, _, _ = jl.lens.apply(jl.lm, prompt, positions=[position], layers=layers)
    rows = {}
    for L in layers:
        logits = lens_logits[L][0]
        rows[L] = {n: float(logits[ids].max()) for n, ids in cid.items()}
    return rows


class EvalRunner:
    """Runs a `StimulusSet` through a fitted `JLensVL` (`jl`) and scores it."""

    def __init__(self, jl):
        self.jl = jl

    def run_item(self, stim_set: StimulusSet, item: Stimulus, *, layers=None, position="answer") -> dict:
        """Score one item. Returns a flat per-item result dict:
        `{id, category, target, distractors, scores, skipped, reason, meta}`
        -- the shape `metrics.aggregate` and `EvalRunner.run` both expect."""
        base = {"id": item.id, "category": item.category, "target": item.target, "meta": item.meta}

        target_ids = self.jl._word_ids(stim_set.concepts[item.target].words)
        if not target_ids:
            return {
                **base,
                "distractors": list(item.distractors),
                "scores": {},
                "skipped": True,
                "reason": "target concept has no single-token surface form",
            }

        scorable_distractors = [
            d for d in item.distractors if self.jl._word_ids(stim_set.concepts[d].words)
        ]
        dropped = [d for d in item.distractors if d not in scorable_distractors]
        concepts = {item.target: stim_set.concepts[item.target].words}
        concepts.update({d: stim_set.concepts[d].words for d in scorable_distractors})

        if item.image:
            scores = self.jl.concept_race(item.image, item.prompt, concepts, layers=layers, position=position)
        else:
            pos = -1 if position == "answer" else position
            scores = score_text_concepts(self.jl, item.prompt, concepts, layers=layers, position=pos)

        reason = f"dropped non-single-token distractors: {dropped}" if dropped else None
        return {
            **base,
            "distractors": scorable_distractors,
            "scores": scores,
            "skipped": False,
            "reason": reason,
        }

    def run(self, stimulus_set: StimulusSet, *, layers=None, position="answer") -> list:
        """Score every item in `stimulus_set`. Returns `list[per-item result]`."""
        return [
            self.run_item(stimulus_set, item, layers=layers, position=position)
            for item in stimulus_set.items
        ]

    def evaluate(self, stimulus_set: StimulusSet, *, layers=None, position="answer", final_layer=None) -> dict:
        """`run` + `metrics.aggregate` in one call. `final_layer` defaults to
        the lens's last fitted layer (`jl.lens.source_layers[-1]`) when a
        lens is attached."""
        results = self.run(stimulus_set, layers=layers, position=position)
        if final_layer is None and getattr(self.jl, "lens", None) is not None:
            final_layer = self.jl.lens.source_layers[-1]
        return aggregate(results, final_layer=final_layer)

    # ---------- synthetic mode: exercise the pipeline without a model ----------
    @staticmethod
    def run_synthetic(stimulus_set: StimulusSet, *, layers=(0, 4, 8, 12, 16, 20), seed=0) -> list:
        """Fabricate plausible per-layer scores for every item, with no model
        involved: the target concept's score ramps up across `layers` while
        each distractor stays flat-ish, plus small deterministic noise
        (seeded per-item so runs are reproducible). This is NOT a real
        result -- it only proves the run/aggregate/report/CLI plumbing works
        end-to-end (see `--synthetic` in `scripts/eval_lens.py`)."""
        import random

        results = []
        n = len(layers)
        for item in stimulus_set.items:
            rng = random.Random(f"{seed}:{item.id}")
            scores = {}
            for i, L in enumerate(layers):
                ramp = (i + 1) / n  # 0 -> 1 across the given layers
                row = {item.target: 2.0 + 8.0 * ramp + rng.uniform(-0.5, 0.5)}
                for d in item.distractors:
                    row[d] = 3.0 + rng.uniform(-1.0, 1.0)
                scores[L] = row
            results.append(
                {
                    "id": item.id,
                    "category": item.category,
                    "target": item.target,
                    "distractors": list(item.distractors),
                    "meta": item.meta,
                    "scores": scores,
                    "skipped": False,
                    "reason": None,
                }
            )
        return results
