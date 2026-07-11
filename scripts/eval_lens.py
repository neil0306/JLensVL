#!/usr/bin/env python
"""CLI: evaluate a fitted JLensVL against a controlled StimulusSet.

Real mode (needs downloaded model weights + a GPU):

    CUDA_VISIBLE_DEVICES=1 python scripts/eval_lens.py \\
        --model Qwen/Qwen3.5-4B --lens lens_qwen35_4b_final.pt \\
        --set data/eval_sets/association_text.jsonl --out report.json

Synthetic mode (no model, no GPU -- fabricates scores to prove the
run/aggregate/report pipeline works end-to-end):

    python scripts/eval_lens.py --synthetic \\
        --set data/eval_sets/association_text.jsonl --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow `python scripts/eval_lens.py` from the repo root without an installed package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jlensvl.eval import EvalRunner, StimulusSet  # noqa: E402
from jlensvl.eval.metrics import aggregate  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True, help="Path to a StimulusSet JSONL file.")
    p.add_argument("--model", default=None, help="HF model id, e.g. Qwen/Qwen3.5-4B (real mode only).")
    p.add_argument("--lens", default=None, help="Path to a saved JacobianLens .pt (real mode only).")
    p.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="Layer indices. Default: the lens's fitted source_layers (real mode) "
             "or (0,4,8,12,16,20) (synthetic mode).",
    )
    p.add_argument("--position", default="answer", help='Read-out position: "answer" or an int index.')
    p.add_argument("--out", default=None, help="Write the JSON report here.")
    p.add_argument(
        "--synthetic", action="store_true",
        help="Fabricate scores instead of loading a model -- proves the eval pipeline runs "
             "end-to-end without weights/GPU. Real mode needs --model + --lens and a GPU "
             "(CUDA_VISIBLE_DEVICES=1); do not use GPU 0.",
    )
    return p


def _fmt(x, nd=2):
    return f"{x:.{nd}f}" if x is not None else "-"


def _print_table(report: dict) -> None:
    header = f"{'category':<22}{'n_items':>8}{'n_scored':>9}{'acc@final':>11}{'mean_fcl':>10}{'mean_margin':>13}"
    print(header)
    print("-" * len(header))

    def row(name, stats):
        print(
            f"{name:<22}{stats['n_items']:>8}{stats['n_scored']:>9}"
            f"{_fmt(stats['accuracy_final_layer']):>11}"
            f"{_fmt(stats['mean_first_correct_layer']):>10}"
            f"{_fmt(stats['mean_peak_margin']):>13}"
        )

    row("__all__", report)
    for cat, stats in sorted(report["by_category"].items()):
        row(cat, stats)


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    stim_set = StimulusSet.from_jsonl(args.set)
    print(f"loaded {args.set}: {len(stim_set.items)} items, {len(stim_set.concepts)} concepts")

    if args.synthetic:
        layers = tuple(args.layers) if args.layers else (0, 4, 8, 12, 16, 20)
        results = EvalRunner.run_synthetic(stim_set, layers=layers)
        report = aggregate(results, final_layer=layers[-1])
        mode = "synthetic (fabricated scores -- NOT a real result, proves the pipeline only)"
    else:
        if not args.model or not args.lens:
            raise SystemExit(
                "real mode needs --model and --lens (or pass --synthetic); "
                "real mode also needs downloaded weights + a GPU other than GPU 0 "
                "(CUDA_VISIBLE_DEVICES=1)"
            )
        from jlensvl import JLensVL

        jl = JLensVL.from_pretrained(args.model, lens=args.lens)
        runner = EvalRunner(jl)
        report = runner.evaluate(stim_set, layers=args.layers, position=args.position)
        mode = f"real ({args.model}, lens={args.lens})"

    print(f"mode: {mode}")
    _print_table(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

    return report


if __name__ == "__main__":
    main()
