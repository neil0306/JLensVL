"""Text J-Lens: watch a concept form before it's spoken (boot -> Italy -> euro),
and compare the true Jacobian lens against the plain logit-lens baseline."""
import os
from jlensvl import JLensVL

MID = os.environ.get("MODEL_DIR", "Qwen/Qwen3.5-4B")
LENS = os.environ.get("LENS", "lens_qwen35_4b_final.pt")

jl = JLensVL.from_pretrained(MID, lens=LENS, multimodal=False)
probe = "Fact: The currency used in the country shaped like a boot is the"

for use_j in (True, False):
    print(f"\n=== {'J-LENS (Jacobian)' if use_j else 'LOGIT-LENS (baseline)'} ===")
    trace = jl.trace(probe, position=-1, k=6, use_jacobian=use_j)
    for layer in sorted(trace):
        if layer % 2 == 0 or layer >= 25:
            print(f"  L{layer:02d} {trace[layer]}")
