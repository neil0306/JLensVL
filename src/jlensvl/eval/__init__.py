"""Quantitative evaluation + controlled-stimuli infrastructure for JLensVL.

    from jlensvl.eval import Concept, Stimulus, StimulusSet, EvalRunner
    from jlensvl.eval import correct_at_layer, first_correct_layer, margin_at_layer, crossover_layer, aggregate

`stimuli` defines the labeled-dataset schema (round-trips through JSONL),
`metrics` is pure per-item/dataset-level scoring math, `runner` drives a
fitted `JLensVL` over a `StimulusSet` and calls `metrics.aggregate`.
"""
from .stimuli import Concept, Stimulus, StimulusSet
from .metrics import (
    aggregate,
    correct_at_layer,
    crossover_layer,
    first_correct_layer,
    margin_at_layer,
)
from .runner import EvalRunner, score_text_concepts

__all__ = [
    "Concept",
    "Stimulus",
    "StimulusSet",
    "correct_at_layer",
    "first_correct_layer",
    "margin_at_layer",
    "crossover_layer",
    "aggregate",
    "EvalRunner",
    "score_text_concepts",
]
