"""VLM J-Lens: feed a photo, get the model's answer, then read what the model is
poised to say at the answer position across depth. The lens often reveals finer
detail than the model's short answer (e.g. 'pug' when it only says 'dog')."""
import os, sys
from jlensvl import JLensVL

MID = os.environ.get("MODEL_DIR", "Qwen/Qwen3.5-4B")
LENS = os.environ.get("LENS", "lens_qwen35_4b_final.pt")
image = sys.argv[1] if len(sys.argv) > 1 else "images/dog.jpg"

jl = JLensVL.from_pretrained(MID, lens=LENS)          # auto-detects the vision tower
print("MODEL says:", repr(jl.describe(image)))

res = jl.trace_image(image, "What is the main subject of this photo? Answer briefly.",
                     layers=[15, 20, 25, 29, 30], k=6)
print("\nJ-Lens @ answer position (concept forming for the reply):")
for L, toks in res["answer"].items():
    print(f"  L{L:02d} {toks}")
