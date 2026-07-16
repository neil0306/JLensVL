"""Fit a vision-tower Jacobian lens for the Qwen3.5-4B ViT (vision face of JLensVL).

For each ViT block l, estimate J_l = E[d h_23 / d h_l] (the transport of that block's residual
into the final block's residual basis), token- and image-averaged, via backprop through the
vision tower -- using the `block_jacobian.fit_block_jacobian` estimator (here its packed
variant, as the Qwen vision forward packs tokens with no batch axis).

Weights (no LLM decoder needed): the (unquantized bf16) vision tower + merger come from a
reconstructed vis_only safetensors; the tied unembed (embed_tokens) comes from the AWQ
snapshot's first shard. See src/jlensvl/vision_lens.py::VisionJLens.from_qwen35.

Run (inside the `jlensvl` conda env, GPU 0 = the 96GB Blackwell):

    CUDA_VISIBLE_DEVICES=0 python examples/10_fit_vision_lens.py

Env knobs: AWQ_SNAPSHOT, VIS_WEIGHTS, OUT (default results/vision_lens), R (channel batch,
default 128), FIT_SIZE (default 448).
"""
import os
import sys
import json
import glob

import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from jlensvl.vision_lens import VisionJLens  # noqa: E402

AWQ = os.environ.get(
    "AWQ_SNAPSHOT",
    "/home/anu/.cache/huggingface/hub/models--QuantTrio--Qwen3.5-4B-AWQ/snapshots/"
    "32c292e3a73afe1138518180b1b6d2868c980ee2")
VIS = os.environ.get("VIS_WEIGHTS", "/home/anu/hf_probe_cache/vis_only/visual.safetensors")
OUT = os.environ.get("OUT", os.path.join(HERE, "..", "results", "vision_lens"))
R = int(os.environ.get("R", "128"))
FIT_SIZE = int(os.environ.get("FIT_SIZE", "448"))
os.makedirs(OUT, exist_ok=True)

# Calibration = the natural object photos at full frame (generic scenes; the J-Lens is a mean
# over patch tokens + images, so a handful of diverse images suffices -- and none of them are
# the composited *test* images used by the validations, so there is no leakage).
INPUTS = os.path.join(OUT, "inputs")
calib = sorted(glob.glob(os.path.join(INPUTS, "*.jpg")))
if not calib:
    raise SystemExit(f"no calibration images in {INPUTS}")
print(f"[vfit] calibration images ({len(calib)}): {[os.path.basename(c) for c in calib]}")

vl = VisionJLens.from_qwen35(awq_snapshot=AWQ, vis_weights=VIS)
print(f"[vfit] loaded vision tower: {vl.n_blocks} blocks, d_vision={vl.d_vision}, "
      f"d_model={vl.d_model}, merge_unit={vl.merge_unit}, R={R}")

lens = vl.fit(calib, R=R, size=FIT_SIZE)
lens_path = os.path.join(OUT, "vision_jacobian_lens.pt")
lens.save(lens_path)
sz = os.path.getsize(lens_path) / 1e6
print(f"\n[vfit] saved lens -> {lens_path} ({sz:.1f} MB)  {lens!r}")

with open(os.path.join(OUT, "fit_meta.json"), "w") as f:
    json.dump({"lens_path": lens_path, "size_mb": round(sz, 1),
               "calibration": [os.path.basename(c) for c in calib],
               "R": R, "fit_size": FIT_SIZE, "d_vision": vl.d_vision,
               "target_block": lens.target_block, **lens.meta}, f, indent=2)
print(f"[vfit] wrote {os.path.join(OUT, 'fit_meta.json')}")
