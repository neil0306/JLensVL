"""JLensVL full stack — run BOTH faces of the J-Lens on one image.

  (A) VISION TOWER : VisionJLens over the Qwen3.5-4B ViT (24 blocks). For each block,
                     decode the merged-patch residuals into the LLM vocab; report where the
                     object localizes, naive vision-logit-lens vs fitted vision-Jacobian-lens.
  (B) LLM DECODER  : JLensVL over the full VLM. The model's own answer, what the decoder is
                     *poised to say* at the answer position across LLM depth, and a concept
                     race between candidate objects.

Both faces read the SAME image, tracing a concept pixels -> ViT depth -> LLM decoder -> word.

Everything is reproducible with no local paths: base model = public Qwen/Qwen3.5-4B
(auto-downloaded); both fitted lenses = TerryYu/JLensVL-lenses (auto-downloaded). A demo
image ships in examples/assets/vision.

Run (GPU 0):
    CUDA_VISIBLE_DEVICES=0 python examples/12_full_stack.py [image.jpg] [object]
    # e.g.  python examples/12_full_stack.py examples/assets/vision/dog.jpg dog
"""
import os
import sys
import json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from _common import MODEL_ID, DEMO_IMAGES, lens_path     # noqa: E402

IMG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DEMO_IMAGES, "dog.jpg")
OBJ = sys.argv[2] if len(sys.argv) > 2 else "dog"
DEV = os.environ.get("JLENSVL_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

report = {"image": IMG, "object": OBJ, "model": MODEL_ID}
print(f"\n{'='*70}\nJLensVL full stack — image={IMG}  object='{OBJ}'\n{'='*70}")

# ============================================================ (A) VISION TOWER
print("\n### (A) VISION TOWER — VisionJLens over the ViT ###")
from jlensvl.vision_lens import VisionJLens
vl = VisionJLens.from_pretrained(MODEL_ID, device=DEV, lens=lens_path("vision"))
print(f"[A] vision tower: {vl.n_blocks} blocks, d_vision={vl.d_vision}, lens={vl.lens!r}")

res_j = vl.read_image(IMG, use_jacobian=True)            # fitted Jacobian lens, all blocks
ids = vl.token_ids([OBJ])
rows, cols = res_j["rows"], res_j["cols"]
if ids:
    # object heatmap peak per block; report where the ViT localizes the object best
    peaks = []
    for L in range(vl.n_blocks):
        obj = res_j["logits"][L][:, ids].max(dim=1).values          # [P]
        peak = int(obj.argmax())
        peaks.append((L, peak // cols, peak % cols, float(obj.max())))
    best = max(peaks, key=lambda t: t[3])
    report["vision"] = {"grid": [rows, cols], "best_block": best[0],
                        "best_peak_rc": [best[1], best[2]]}
    print(f"[A] '{OBJ}' localizes strongest at ViT block L{best[0]}, "
          f"merged-patch (row={best[1]}, col={best[2]}) of the {rows}x{cols} grid")
else:
    print(f"[A] '{OBJ}' has no single-token id; skipping vision object-scoring")
    report["vision"] = {"grid": [rows, cols]}
print("[A] (rigorous naive-vs-Jacobian V1/V2/V3 numbers: run examples/11)")

del vl
if DEV.startswith("cuda"):
    torch.cuda.empty_cache()

# ============================================================ (B) LLM DECODER
print("\n### (B) LLM DECODER — JLensVL over the full VLM ###")
from jlensvl import JLensVL
jl = JLensVL.from_pretrained(MODEL_ID, lens=lens_path("llm"), device=DEV)
print(f"[B] model={MODEL_ID}  n_layers={jl.n_layers}  d_model={jl.d_model}  "
      f"image_token_id={jl.image_token_id}")

Q = "What is the main subject of this photo? Answer briefly."
answer = jl.describe(IMG, Q)
print(f"[B] MODEL says: {answer!r}")
report["llm"] = {"question": Q, "answer": answer}

LAYERS = [15, 20, 25, 29, 30]
trace = jl.trace_image(IMG, Q, layers=LAYERS, k=6)
print("[B] J-Lens @ answer position (what the decoder is poised to say), across LLM depth:")
for L, toks in trace["answer"].items():
    print(f"      L{L:02d} {toks}")
report["llm"]["poised_at_answer"] = {str(L): t for L, t in trace["answer"].items()}

race_concepts = {OBJ: [OBJ], "cat": ["cat"], "car": ["car"], "person": ["person", "man", "woman"]}
race = jl.concept_race(IMG, Q, race_concepts, layers=LAYERS)
print("[B] concept race (max-logit per class) across LLM depth:")
print("      block : " + "  ".join(f"{k:>8}" for k in race_concepts))
for L in LAYERS:
    print(f"      L{L:02d}   : " + "  ".join(f"{race[L][k]:8.2f}" for k in race_concepts))
report["llm"]["concept_race"] = {str(L): race[L] for L in LAYERS}

out = os.path.join(HERE, "..", "results", "full_stack_report.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[done] wrote {out}")
print(f"\nSUMMARY: ViT localizes '{OBJ}' strongest at block "
      f"L{report['vision'].get('best_block','?')}; LLM answers {answer!r}")
