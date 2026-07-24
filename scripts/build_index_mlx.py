"""Build the JLensVL retrieval index natively on Apple Silicon via MLX.

The residual-stream capture (``MLXEncoder``) and the MLX query-path proposer
(``propose_concept_mlx``) now live in the reusable module
``jlensvl.mlx_encoder`` — the same code the Prompt Helper Studio app uses at
runtime, so index-build and query encoding can never drift apart. This script is
just the CLI around them: load an MLX Qwen3.5, build the index over a corpus via
``build_index``'s ``encode_fn`` seam (no torch model / JLens engine needed), and
run a few ``propose_concept`` smoke checks.
"""
from __future__ import annotations
import argparse, sys, time, glob, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from mlx_lm import load

from jlensvl.retrieval_lens import build_index, RetrievalIndex, default_layers
from jlensvl.mlx_encoder import MLXEncoder, propose_concept_mlx

MODEL = "mlx-community/Qwen3.5-4B-MLX-8bit"


def load_corpus(root, include_vg=False, max_lines=None):
    files = sorted(glob.glob(os.path.join(root, "*.txt")))
    corpus = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.startswith("vg_") and not include_vg:
            continue
        tag = "vg" if stem.startswith("vg_") else f"corpus:{stem}"
        with open(f, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if max_lines:
            lines = lines[:max_lines]
        corpus.extend((ln, tag) for ln in lines)
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="retrieval_index_mlx.pt")
    ap.add_argument("--include-vg", action="store_true")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--corpus-root", default="data/retrieval_corpus")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--verify-only", default=None,
                    help="path to an existing index to only run propose checks")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    layers = default_layers(32)
    print("layers:", layers, "dtype:", args.dtype)

    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"model loaded in {time.time()-t0:.1f}s")
    enc = MLXEncoder(model, tokenizer, layers, args.max_length)

    if args.verify_only:
        index = RetrievalIndex.load(args.verify_only)
    else:
        corpus = load_corpus(args.corpus_root, args.include_vg, args.max_lines)
        print(f"corpus: {len(corpus)} sentences "
              f"(include_vg={args.include_vg}, max_lines={args.max_lines})")
        t1 = time.time()
        index = build_index(None, corpus, layers=layers, dtype=dtype,
                            encode_fn=enc.encode_fn, show_progress=True)
        dt = time.time() - t1
        print(f"built in {dt:.1f}s  |  {index!r}")
        index.save(args.out)
        sz = os.path.getsize(args.out) / 1e6
        print(f"saved {args.out}  ({sz:.1f} MB)")

    index.to("cpu")
    print("\n=== propose_concept checks (MLX query) ===")
    checks = [("fighting", "{}"), ("打架", "{}"), ("phone", "{}"),
              ("手机", "{}"), ("helmet", "The worker is not wearing a {}")]
    for concept, tmpl in checks:
        res = propose_concept_mlx(index, enc, concept, k=8, template=tmpl)
        words = [f"{r['word']}({r['score']:.2f})" for r in res]
        print(f"  {concept!r:12} tmpl={tmpl!r:35} -> {words}")


if __name__ == "__main__":
    main()
