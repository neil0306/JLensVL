#!/usr/bin/env python
"""P3 — the real task-accuracy bridge.

Tests the core prompt-helper claim:  does the OBSERVATIONAL lens-margin the
helper assigns to a prompt VARIANT predict the model's REAL accuracy on a
labeled task?  If the variant the helper ranks #1 also gets the highest real
accuracy (and lens-margin correlates with accuracy across variants), the
tool's advice translates to real performance — using only the model's own
correctness on labeled data, no external judge.

For each prompt variant (same task, different wording) this harness:

  (a) OBSERVATIONAL lens-margin (forward-only, NO ground-truth used) —
      renders every labeled example through the model's REAL chat template
      (`PromptHelper.trace_rendered`) and reads the three verdict senses
      {CONFIRMED, POSSIBLE, CLEAR} at the answer position.  The per-example
      margin is the model's *verdict decisiveness* = (top verdict sense −
      runner-up verdict sense).  The variant's `lens_margin` is the mean of
      those over all examples.  This is exactly the "how decisively is the
      prompt steering the model into the answer space" signal the helper
      exposes — computed with no labels.

  (b) REAL accuracy — actually generates the model's answer for every labeled
      example under that variant's wording, parses the verdict word, and
      scores it against the ground-truth label.

Then it correlates per-variant {lens_margin, accuracy} (Pearson + Spearman),
reports which variant each METHOD picks as best (helper => argmax lens_margin;
reality => argmax accuracy), and whether they agree.

Everything that touches the model/GPU lives behind the real path; `--self-test`
exercises all the non-model logic (arg parsing, task/variant loading, message
rendering, the correlation math on FAKE per-variant numbers) with no weights
and no GPU, so the wiring can be verified on CPU before the GPU run.

Real run (GPU1 — never GPU 0):

    CUDA_VISIBLE_DEVICES=1 JLENSVL_MODEL_PATH=/home/anu/qwen35_4b_dl \\
        python scripts/p3_accuracy_bridge.py \\
        --model /home/anu/qwen35_4b_dl \\
        --lens lens_qwen35_4b_final.pt \\
        --task data/eval_sets/p3_task.jsonl \\
        --variants data/eval_sets/p3_prompt_variants.json \\
        --out reports/p3_accuracy_bridge.json

CPU dry-run (no model, proves the plumbing):

    python scripts/p3_accuracy_bridge.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

# allow `python scripts/p3_accuracy_bridge.py` from the repo root uninstalled
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

LABELS = ["CONFIRMED", "POSSIBLE", "CLEAR"]

# Verdict senses for the J-Lens readout: each tier's surface word forms.
# (Single-token scoring is handled inside PromptHelper via `_word_ids`.)
VERDICT_SENSES = {
    "CONFIRMED": ["CONFIRMED", "Confirmed", "confirmed"],
    "POSSIBLE": ["POSSIBLE", "Possible", "possible"],
    "CLEAR": ["CLEAR", "Clear", "clear"],
}


# --------------------------------------------------------------------------
# data loading (pure, no model)
# --------------------------------------------------------------------------
def load_task(path):
    """Read the labeled task JSONL. Skips an optional `_header` line.
    Returns list of {id, text, label}."""
    items = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        if d.get("_header"):
            continue
        if d["label"] not in LABELS:
            raise ValueError(f"item {d.get('id')!r} has unknown label {d['label']!r}; "
                             f"expected one of {LABELS}")
        items.append({"id": d["id"], "text": d["text"], "label": d["label"]})
    if not items:
        raise ValueError(f"{path}: no labeled items found")
    return items


def load_variants(path):
    """Read the prompt-variants JSON. Returns {name: {system, user_template, ...}}."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    variants = doc.get("variants", doc)
    out = {}
    for name, cfg in variants.items():
        if name.startswith("_"):
            continue
        if "user_template" not in cfg or "{text}" not in cfg["user_template"]:
            raise ValueError(f"variant {name!r} must have a user_template containing '{{text}}'")
        out[name] = cfg
    if not out:
        raise ValueError(f"{path}: no variants found")
    return out


def build_messages(variant_cfg, text):
    """One task example -> chat messages for this variant's wording."""
    msgs = []
    if variant_cfg.get("system"):
        msgs.append({"role": "system", "content": variant_cfg["system"]})
    msgs.append({"role": "user",
                 "content": variant_cfg["user_template"].replace("{text}", text)})
    return msgs


# --------------------------------------------------------------------------
# correlation / ranking math (pure, no model) — verified by --self-test
# --------------------------------------------------------------------------
def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _rank(vals):
    """Average (tie-corrected) ranks, ascending."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def correlate_variants(per_variant):
    """per_variant: {name: {"lens_margin": float, "accuracy": float, ...}}.
    Returns the bridge verdict: correlations + which variant each method picks."""
    names = list(per_variant)
    margins = [per_variant[n]["lens_margin"] for n in names]
    accs = [per_variant[n]["accuracy"] for n in names]

    helper_best = max(names, key=lambda n: per_variant[n]["lens_margin"])
    real_best = max(names, key=lambda n: per_variant[n]["accuracy"])

    # rank agreement: do the two orderings agree on the ordering of variants?
    margin_rank = {n: r for n, r in zip(names, _rank(margins))}
    acc_rank = {n: r for n, r in zip(names, _rank(accs))}

    pearson = _pearson(margins, accs)
    spearman = _spearman(margins, accs)
    return {
        "n_variants": len(names),
        "pearson_lensmargin_vs_accuracy": pearson,
        "spearman_lensmargin_vs_accuracy": spearman,
        "helper_pick": helper_best,
        "real_best": real_best,
        "helper_pick_is_real_best": helper_best == real_best,
        "helper_pick_accuracy": per_variant[helper_best]["accuracy"],
        "real_best_accuracy": per_variant[real_best]["accuracy"],
        "accuracy_gap_of_helper_pick": per_variant[real_best]["accuracy"]
        - per_variant[helper_best]["accuracy"],
        "margin_rank": margin_rank,
        "accuracy_rank": acc_rank,
        "claim_supported": bool(
            (pearson is not None and pearson > 0)
            and (helper_best == real_best
                 or per_variant[real_best]["accuracy"]
                 - per_variant[helper_best]["accuracy"] <= 0.05)
        ),
    }


# --------------------------------------------------------------------------
# model path (GPU) — only imported/executed in the real run
# --------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TIER_JSON_RE = re.compile(r'"tier"\s*:\s*"(confirmed|possible|clear)"', re.IGNORECASE)


def _parse_verdict(text):
    """Map a free-form model answer to one of LABELS, else None.

    Robust to Qwen3.5 thinking output: any ``<think>...</think>`` block is
    stripped first. We then prefer an explicit JSON verdict
    (``{"tier": "CONFIRMED"|...}``, case-insensitive); failing that we fall
    back to the FIRST bare tier keyword (word-boundary matched so "clearly"
    does not spuriously match CLEAR)."""
    if not text:
        return None
    text = _THINK_RE.sub(" ", text)
    m = _TIER_JSON_RE.search(text)
    if m:
        return m.group(1).upper()
    up = text.upper()
    best = None
    best_pos = len(up) + 1
    for lab in LABELS:
        m = re.search(r"\b" + lab + r"\b", up)
        if m and m.start() < best_pos:
            best, best_pos = lab, m.start()
    return best


def _observational_margin(helper, variant_cfg, examples, *, layer):
    """Mean verdict-decisiveness lens-margin for one variant (NO labels used).

    Per example: render through the real chat template, read the three verdict
    senses at the answer position, margin = top verdict − runner-up verdict.
    Returns (mean_margin, per_example_list)."""
    per = []
    for ex in examples:
        msgs = build_messages(variant_cfg, ex["text"])
        # Read the verdict senses with thinking OFF so the answer position is
        # "poised to answer" — the SAME regime we generate under below.
        tr = helper.trace_rendered(msgs, senses=VERDICT_SENSES, layer=layer,
                                   enable_thinking=False)
        sc = tr["senses"] or {}
        vals = sorted((v for v in sc.values() if v is not None), reverse=True)
        if len(vals) < 2:
            continue
        top_sense = max(sc, key=lambda k: (sc[k] if sc[k] is not None else -1e30))
        per.append({"id": ex["id"], "senses": sc,
                    "top_sense": top_sense, "margin": vals[0] - vals[1]})
    mean = sum(p["margin"] for p in per) / len(per) if per else float("nan")
    return mean, per


def _real_accuracy(jl, helper, variant_cfg, examples, *, max_new_tokens):
    """Generate the model's answer for every example under this variant and
    score it against the ground-truth label. Returns (accuracy, per_example).

    Generation is rendered with thinking OFF (via PromptHelper._render, which
    genuinely honors enable_thinking=False on this template) so the model emits
    the verdict word directly instead of spending the token budget on a
    ``Thinking Process: ...`` preamble that gets truncated before the answer."""
    import torch
    tok = jl.tok
    per = []
    n_correct = 0
    for ex in examples:
        msgs = build_messages(variant_cfg, ex["text"])
        rendered = helper._render(msgs, add_generation_prompt=True,
                                  enable_thinking=False)
        inputs = tok(rendered, return_tensors="pt").to(jl.model.device)
        with torch.no_grad():
            gen = jl.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False)
        answer = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
        pred = _parse_verdict(answer)
        ok = (pred == ex["label"])
        n_correct += int(ok)
        per.append({"id": ex["id"], "label": ex["label"], "pred": pred,
                    "correct": ok, "answer": answer[:200]})
    acc = n_correct / len(examples) if examples else float("nan")
    return acc, per


def run_real(args, task, variants):
    """Load the model + lens on GPU and produce the full bridge result."""
    from jlensvl import JLensVL
    from jlensvl.prompt_helper import PromptHelper

    if not args.model or not args.lens:
        raise SystemExit("real mode needs --model and --lens (or pass --self-test)")

    print(f"[p3] loading model={args.model} lens={args.lens} device={args.device}",
          flush=True)
    jl = JLensVL.from_pretrained(args.model, lens=args.lens, device=args.device)
    helper = PromptHelper(jl)

    examples = task[: args.limit] if args.limit else task
    print(f"[p3] {len(examples)} labeled examples, {len(variants)} variants", flush=True)

    per_variant = {}
    details = {}
    for name, cfg in variants.items():
        print(f"[p3] variant [{name}] — observational lens-margin ...", flush=True)
        margin, margin_per = _observational_margin(helper, cfg, examples, layer=args.layer)
        print(f"[p3] variant [{name}] — real generation accuracy ...", flush=True)
        acc, acc_per = _real_accuracy(jl, helper, cfg, examples,
                                      max_new_tokens=args.max_new_tokens)
        per_variant[name] = {"lens_margin": margin, "accuracy": acc,
                             "n_examples": len(examples)}
        details[name] = {"margin_per_example": margin_per, "accuracy_per_example": acc_per}
        print(f"[p3]   [{name}] lens_margin={margin:.3f}  accuracy={acc:.3f}", flush=True)

    bridge = correlate_variants(per_variant)
    return {"mode": "real", "model": str(args.model), "lens": str(args.lens),
            "layer": args.layer, "n_examples": len(examples),
            "per_variant": per_variant, "bridge": bridge, "details": details}


# --------------------------------------------------------------------------
# CPU self-test — no model, no GPU
# --------------------------------------------------------------------------
def run_self_test(args):
    """Exercise every non-model code path and assert the correlation math is
    wired correctly on FAKE per-variant numbers."""
    print("[self-test] loading task + variants (real files, no model)...")
    task = load_task(args.task)
    variants = load_variants(args.variants)
    print(f"[self-test]   task: {len(task)} labeled items "
          f"({sum(1 for t in task if t['label'] == 'CONFIRMED')} CONFIRMED / "
          f"{sum(1 for t in task if t['label'] == 'POSSIBLE')} POSSIBLE / "
          f"{sum(1 for t in task if t['label'] == 'CLEAR')} CLEAR)")
    print(f"[self-test]   variants: {list(variants)}")

    # message rendering works for every variant on a sample example
    sample = task[0]
    for name, cfg in variants.items():
        msgs = build_messages(cfg, sample["text"])
        assert msgs[-1]["role"] == "user" and sample["text"] in msgs[-1]["content"], name
        assert "{text}" not in msgs[-1]["content"], f"placeholder unfilled in {name}"
    print(f"[self-test]   build_messages OK for all {len(variants)} variants "
          f"(sample id={sample['id']})")

    # verdict parser
    assert _parse_verdict("The verdict is CONFIRMED.") == "CONFIRMED"
    assert _parse_verdict("lit cigarette -> confirmed") == "CONFIRMED"
    assert _parse_verdict("I think this is Possible, maybe.") == "POSSIBLE"
    assert _parse_verdict("clearly CLEAR") == "CLEAR"
    assert _parse_verdict("no idea") is None
    # word-boundary: a bare "clearly" must NOT be read as CLEAR
    assert _parse_verdict("this is clearly a cigarette\nCONFIRMED") == "CONFIRMED"
    # explicit JSON verdict wins (case-insensitive)
    assert _parse_verdict('{"tier": "Possible", "why": "faint haze"}') == "POSSIBLE"
    # a <think> block is stripped before parsing
    assert _parse_verdict("<think>maybe clear? no, cigarette</think>\nCONFIRMED") == "CONFIRMED"
    print("[self-test]   _parse_verdict OK")

    # correlation math on FAKE per-variant numbers where the ranking is known:
    # margin ascending == accuracy ascending -> perfect positive correlation.
    fake = {
        "terse":   {"lens_margin": 0.5, "accuracy": 0.40},
        "neutral": {"lens_margin": 1.5, "accuracy": 0.55},
        "role":    {"lens_margin": 2.0, "accuracy": 0.60},
        "verbose": {"lens_margin": 2.5, "accuracy": 0.70},
        "reasoned":{"lens_margin": 3.0, "accuracy": 0.75},
        "native":  {"lens_margin": 4.0, "accuracy": 0.90},
    }
    b = correlate_variants(fake)
    # monotone (not exactly linear) -> Spearman is exactly 1.0, Pearson very high
    assert abs(b["spearman_lensmargin_vs_accuracy"] - 1.0) < 1e-9, b
    assert b["pearson_lensmargin_vs_accuracy"] > 0.95, b
    assert b["helper_pick"] == "native" and b["real_best"] == "native"
    assert b["helper_pick_is_real_best"] and b["claim_supported"]
    print(f"[self-test]   correlate_variants (monotone case): "
          f"pearson={b['pearson_lensmargin_vs_accuracy']:.3f} "
          f"spearman={b['spearman_lensmargin_vs_accuracy']:.3f} "
          f"helper_pick={b['helper_pick']} -> claim_supported={b['claim_supported']}")

    # anti-correlated case: helper's pick is the WORST -> claim NOT supported.
    fake2 = {n: {"lens_margin": v["lens_margin"],
                 "accuracy": 1.0 - v["accuracy"]} for n, v in fake.items()}
    b2 = correlate_variants(fake2)
    assert b2["pearson_lensmargin_vs_accuracy"] < 0
    assert not b2["claim_supported"]
    print(f"[self-test]   correlate_variants (anti-correlated case): "
          f"pearson={b2['pearson_lensmargin_vs_accuracy']:.3f} "
          f"claim_supported={b2['claim_supported']} (expected False)")

    # tie / near-miss tolerance: helper pick != top acc but within 5% -> supported
    fake3 = {
        "a": {"lens_margin": 3.0, "accuracy": 0.80},
        "b": {"lens_margin": 2.0, "accuracy": 0.82},
        "c": {"lens_margin": 1.0, "accuracy": 0.50},
    }
    b3 = correlate_variants(fake3)
    assert b3["helper_pick"] == "a" and b3["real_best"] == "b"
    assert b3["claim_supported"], b3  # 0.02 gap <= 0.05 tolerance and pearson>0
    print(f"[self-test]   near-miss tolerance OK "
          f"(helper_pick=a acc=0.80 vs real_best=b acc=0.82, gap<=0.05 -> supported)")

    print("\n[self-test] ALL CHECKS PASSED — non-model plumbing is wired correctly.")
    print("[self-test] The model/GPU path (_observational_margin + _real_accuracy) "
          "was NOT run (needs weights + GPU1).")
    return {"mode": "self-test", "task_items": len(task),
            "variants": list(variants), "checks": "passed"}


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="data/eval_sets/p3_task.jsonl",
                   help="Labeled task JSONL (id/text/label).")
    p.add_argument("--variants", default="data/eval_sets/p3_prompt_variants.json",
                   help="Prompt-variants JSON.")
    p.add_argument("--model", default=os.environ.get("JLENSVL_MODEL_PATH"),
                   help="HF model id or local weights dir (real mode). "
                        "Defaults to $JLENSVL_MODEL_PATH.")
    p.add_argument("--lens", default="lens_qwen35_4b_final.pt",
                   help="Path to a saved JacobianLens .pt (real mode).")
    p.add_argument("--layer", type=int, default=None,
                   help="J-Lens layer for the observational margin "
                        "(default: the lens's own second-to-last fitted layer).")
    p.add_argument("--device", default="auto", help='"auto"/"cuda"/"cuda:0"/"cpu".')
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Generation budget per example for the real-accuracy pass. "
                        "With thinking OFF the verdict word is emitted immediately, "
                        "so 64 is ample headroom (even for the reason-then-answer "
                        "variant).")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N labeled examples (quick smoke run).")
    p.add_argument("--out", default=None, help="Write the JSON result here.")
    p.add_argument("--self-test", action="store_true",
                   help="CPU dry-run: load task+variants, render messages, and check "
                        "the correlation math on FAKE numbers. No model, no GPU.")
    return p


def _print_summary(result):
    if result.get("mode") == "self-test":
        return
    print("\n" + "=" * 64)
    print("P3 ACCURACY BRIDGE — per variant")
    print(f"{'variant':<12}{'lens_margin':>13}{'accuracy':>11}")
    print("-" * 36)
    for name, r in result["per_variant"].items():
        print(f"{name:<12}{r['lens_margin']:>13.3f}{r['accuracy']:>11.3f}")
    b = result["bridge"]
    print("-" * 36)
    print(f"pearson(lens_margin, accuracy)  = {b['pearson_lensmargin_vs_accuracy']}")
    print(f"spearman(lens_margin, accuracy) = {b['spearman_lensmargin_vs_accuracy']}")
    print(f"helper picks : [{b['helper_pick']}] (acc={b['helper_pick_accuracy']:.3f})")
    print(f"real best    : [{b['real_best']}] (acc={b['real_best_accuracy']:.3f})")
    print(f"helper pick == real best?  {b['helper_pick_is_real_best']}")
    print(f"CLAIM SUPPORTED: {b['claim_supported']}")
    print("=" * 64)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        result = run_self_test(args)
    else:
        task = load_task(args.task)
        variants = load_variants(args.variants)
        result = run_real(args, task, variants)
        _print_summary(result)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
