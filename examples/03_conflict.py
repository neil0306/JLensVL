"""Contradictory image+text: a DOG photo, but the text claims it's a CAT.
Watch the visual concept override the textual assertion, layer by layer."""
import os, sys
from jlensvl import JLensVL

MID = os.environ.get("MODEL_DIR", "Qwen/Qwen3.5-4B")
LENS = os.environ.get("LENS", "lens_qwen35_4b_final.pt")
image = sys.argv[1] if len(sys.argv) > 1 else "images/dog.jpg"

jl = JLensVL.from_pretrained(MID, lens=LENS)
question = "This image clearly shows a cat. What animal is in this image? Answer with one word."
print("MODEL says:", repr(jl.describe(image, question, max_new_tokens=8)))

race = jl.concept_race(image, question,
                       {"dog": ["dog", "dogs", "puppy", "pug"], "cat": ["cat", "cats", "kitten"]},
                       layers=list(range(12, 31)))
print("\n layer :   dog     cat    winner")
for L in sorted(race):
    d, c = race[L]["dog"], race[L]["cat"]
    print(f"  L{L:02d}  : {d:6.2f}  {c:6.2f}   {'DOG' if d > c else 'cat'} (Δ{d-c:+.2f})")
