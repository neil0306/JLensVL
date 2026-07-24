#!/usr/bin/env python
"""Assemble the corpus (VG generic phrases + domain scenario prompts) and build
+ save a `RetrievalIndex` for the retrieval lens.

This is the **GPU build step** — it loads the real model and runs a forward pass
per corpus sentence. It is intentionally NOT run automatically; invoke it
explicitly on a free GPU. Example (GPU 0 is off-limits on this box):

    CUDA_VISIBLE_DEVICES=1 python scripts/build_retrieval_index.py \
        --model /home/anu/qwen35_4b_dl --scenario \
        --out retrieval_index_qwen35_4b.pt --vg-limit 8000

(--vg-path defaults to the committed data/retrieval_corpus/vg_phrases_20k.txt.)

Corpus sources are TAGGED (source="vg" / source="scenario:<id>") so retrieval can
report and weight by provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# --- make src/ + the jacobian-lens engine importable when run from the repo ---
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", _ROOT.parent / "J-space-test" / "jacobian-lens"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCENARIO_DIR = "/usr/local/visual_llm_complete/server/prompt_scenarios/"
SCENARIO_HOST = "m4max"
# Deterministic 20k-line VG sample committed to the repo; used as the default
# --vg-path so a real build needs no external clone.
DEFAULT_VG_PATH = _ROOT / "data" / "retrieval_corpus" / "vg_phrases_20k.txt"

# Built-in placeholder used only if no VG file is reachable (noted, never blocks).
_PLACEHOLDER_VG = [
    "a red car parked on the street", "a person wearing a hard hat",
    "a wet floor with a warning sign", "a cracked concrete wall",
    "an open flame near flammable material", "a worker on a tall ladder",
    "a spilled chemical container", "a broken glass bottle on the ground",
    "a fire extinguisher mounted on the wall", "a group of people in a meeting",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


def load_vg_corpus(vg_path: str | None, limit: int | None,
                   allow_placeholder: bool = False) -> list[tuple[str, str]]:
    """VG generic phrases, tagged source='vg'.

    Fail-closed: if `vg_path` is missing/unreadable we do NOT silently ship a
    handful of placeholder phrases as if it were a real index — we raise, unless
    `allow_placeholder` is set (a test-only escape hatch)."""
    if vg_path and Path(vg_path).exists():
        lines: list[str] = []
        with open(vg_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    lines.append(s)
                if limit and len(lines) >= limit:
                    break
        return [(s, "vg") for s in lines]
    if allow_placeholder:
        print(f"[warn] VG file {vg_path!r} not found — using "
              f"{len(_PLACEHOLDER_VG)} built-in PLACEHOLDER phrases "
              "(--allow-placeholder). This is NOT a production index.",
              file=sys.stderr)
        return [(s, "vg") for s in _PLACEHOLDER_VG]
    raise SystemExit(
        f"[error] VG file {vg_path!r} not found. Pass --vg-path to a real "
        f"phrase file (default: {DEFAULT_VG_PATH}), or --allow-placeholder for "
        "a throwaway test index, or --no-vg to skip VG entirely.")


def _remote(host: str, remote_cmd: str, timeout: int = 30) -> str:
    """Run `remote_cmd` (already remote-shell-safe) on `host` over ssh."""
    return subprocess.check_output(["ssh", host, remote_cmd], text=True,
                                   timeout=timeout)


def load_scenario_corpus(host: str = SCENARIO_HOST,
                         directory: str = SCENARIO_DIR,
                         strict: bool = True) -> list[tuple[str, str]]:
    """Fetch the production scenario JSONs over ssh, extract every string in
    systemPrompts + userInstructions, split long strings to sentences, tag
    source='scenario:<id>'.

    Fail-closed by default (`strict=True`): a failed ssh/list, or any file we
    can't read/parse, raises rather than silently dropping domain data. All
    remote paths are `shlex.quote`-escaped (no injection via odd filenames such
    as 'Facility Management.json')."""
    out: list[tuple[str, str]] = []
    dir_q = shlex.quote(directory)
    try:
        names = _remote(host, f"ls -1 {dir_q}").splitlines()
    except Exception as exc:  # noqa: BLE001
        msg = f"scenario listing failed on {host}:{directory} ({exc})"
        if strict:
            raise SystemExit(f"[error] {msg}")
        print(f"[warn] {msg}; skipping scenarios.", file=sys.stderr)
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        remote_path = directory + name  # directory ends with '/'
        try:
            raw = _remote(host, f"cat {shlex.quote(remote_path)}")
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            msg = f"could not read/parse scenario {name!r}: {exc}"
            if strict:
                raise SystemExit(f"[error] {msg}")
            print(f"[warn] {msg}", file=sys.stderr)
            continue
        sid = data.get("id") or Path(name).stem
        tag = f"scenario:{sid}"
        for field in ("systemPrompts", "userInstructions"):
            for s in (data.get(field) or []):
                for sent in _split_sentences(s):
                    out.append((sent, tag))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("JLENSVL_MODEL_PATH",
                    "/home/anu/qwen35_4b_dl"), help="model dir or HF id")
    ap.add_argument("--out", default="retrieval_index_qwen35_4b.pt")
    ap.add_argument("--vg-path", default=str(DEFAULT_VG_PATH),
                    help="path to vg_phrases.txt / concepts.txt (one phrase per "
                         f"line); default = committed 20k sample {DEFAULT_VG_PATH}")
    ap.add_argument("--vg-limit", type=int, default=8000,
                    help="cap on VG phrases loaded (0 = all)")
    ap.add_argument("--allow-placeholder", action="store_true",
                    help="if --vg-path is missing, build from a tiny built-in "
                         "placeholder list instead of erroring (test only)")
    ap.add_argument("--scenario", action="store_true",
                    help="also fetch domain scenario prompts over ssh")
    ap.add_argument("--allow-missing-scenarios", action="store_true",
                    help="with --scenario, warn-and-continue instead of erroring "
                         "when the ssh fetch fails or a file can't be parsed")
    ap.add_argument("--no-vg", action="store_true", help="skip the VG source")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="index layers (block-output indices); default = auto by depth")
    ap.add_argument("--reservoir-cap", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total corpus sentences (small test build)")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="index storage dtype")
    ap.add_argument("--device", default=os.environ.get("JLENSVL_DEVICE", "cuda"),
                    help="model device; GPU 0 is off-limits on this box")
    args = ap.parse_args()

    # --- assemble corpus (tagged) ---
    corpus: list[tuple[str, str]] = []
    if not args.no_vg:
        corpus += load_vg_corpus(
            args.vg_path, None if args.vg_limit == 0 else args.vg_limit,
            allow_placeholder=args.allow_placeholder)
    if args.scenario:
        scen = load_scenario_corpus(strict=not args.allow_missing_scenarios)
        if not scen and not args.allow_missing_scenarios:
            sys.exit("[error] --scenario requested but no scenario sentences "
                     "were retrieved (use --allow-missing-scenarios to proceed)")
        corpus += scen
        print(f"[info] scenario sentences: {len(scen)}")
    if args.limit:
        corpus = corpus[: args.limit]
    if not corpus:
        sys.exit("empty corpus — nothing to build")
    print(f"[info] corpus size: {len(corpus)} sentences")

    # --- load model + build (imports here so --help needs no torch) ---
    import torch
    from jlensvl import JLensVL
    from jlensvl.retrieval_lens import build_index

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[info] loading {args.model} on {args.device} (bf16 weights)")
    jl = JLensVL.from_pretrained(args.model, lens=None, device=args.device)
    index = build_index(jl, corpus, layers=args.layers,
                        reservoir_cap=args.reservoir_cap, dtype=dtype)
    print(f"[info] built {index!r}")
    index.save(args.out)
    print(f"[info] saved -> {args.out}")


if __name__ == "__main__":
    main()
