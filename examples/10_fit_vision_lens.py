"""Fit a vision-tower Jacobian lens for the Qwen3.5-4B ViT (vision face of JLensVL).

For each ViT block l, estimate J_l = E[d h_23 / d h_l] (the transport of that block's residual
into the final block's residual basis), token- and image-averaged, via backprop through the
vision tower -- using the `block_jacobian.fit_block_jacobian` estimator (its packed variant,
as the Qwen vision forward packs tokens with no batch axis).

You do NOT need to run this to use the lens: a pre-fitted lens ships on HF
(TerryYu/JLensVL-lenses, auto-downloaded by examples 11/12). Run this only to re-fit.

Only the vision tower + merger + tied unembed are loaded — from a single public checkpoint
(Qwen/Qwen3.5-4B); the LLM decoder is never materialised.

Run (GPU 0):
    CUDA_VISIBLE_DEVICES=0 python examples/10_fit_vision_lens.py

Env knobs: JLENSVL_MODEL (default Qwen/Qwen3.5-4B), CALIB_DIR (default examples/assets/vision),
OUT (default results/vision_lens), R (channel batch, default 128), FIT_SIZE (default 448).
"""
import os
import sys
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from _common import MODEL_ID, DEMO_IMAGES               # noqa: E402
from jlensvl.vision_lens import VisionJLens             # noqa: E402

CALIB_DIR = os.environ.get("CALIB_DIR", DEMO_IMAGES)
OUT = os.environ.get("OUT", os.path.join(HERE, "..", "results", "vision_lens"))
R = int(os.environ.get("R", "128"))
FIT_SIZE = int(os.environ.get("FIT_SIZE", "448"))
os.makedirs(OUT, exist_ok=True)

# Calibration = a handful of diverse object photos at full frame. The J-Lens is a mean over
# patch tokens + images, so a small set suffices; none are the composited *test* images the
# validations (example 11) use, so there is no leakage.
calib = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
if not calib:
    raise SystemExit(f"no calibration images in {CALIB_DIR}")
print(f"[vfit] calibration images ({len(calib)}): {[os.path.basename(c) for c in calib]}")

vl = VisionJLens.from_pretrained(MODEL_ID)
print(f"[vfit] loaded vision tower from {MODEL_ID}: {vl.n_blocks} blocks, d_vision={vl.d_vision}, "
      f"d_model={vl.d_model}, merge_unit={vl.merge_unit}, R={R}")

lens = vl.fit(calib, R=R, size=FIT_SIZE)
lens_path = os.path.join(OUT, "vision_jacobian_lens.pt")
lens.save(lens_path)
sz = os.path.getsize(lens_path) / 1e6
print(f"\n[vfit] saved lens -> {lens_path} ({sz:.1f} MB)  {lens!r}")

with open(os.path.join(OUT, "fit_meta.json"), "w") as f:
    json.dump({"lens_path": lens_path, "size_mb": round(sz, 1), "model": MODEL_ID,
               "calibration": [os.path.basename(c) for c in calib],
               "R": R, "fit_size": FIT_SIZE, "d_vision": vl.d_vision,
               "target_block": lens.target_block, **lens.meta}, f, indent=2)
print(f"[vfit] wrote {os.path.join(OUT, 'fit_meta.json')}")
