"""Prompt helper: read what a prompt is poised to make the model say, and rank
alternative phrasings by how well they steer to an intended sense. Forward-only."""
import os
from jlensvl import JLensVL, PromptHelper

MID = os.environ.get("MODEL_DIR", "Qwen/Qwen3.5-4B")
LENS = os.environ.get("LENS", "lens_qwen35_4b_final.pt")

jl = JLensVL.from_pretrained(MID, lens=LENS, multimodal=False)
ph = PromptHelper(jl)

# 1) what is a prompt poised to say?
for p in ["Java is a", "On the map of Indonesia, Java is a"]:
    r = ph.poised(p)
    print(f"{p!r:45s} -> poised {r['top1']!r} (margin {r['margin']:+.2f})  {r['tokens']}")

# 2) rank phrasings by how well they elicit the INTENDED sense
print("\nRank phrasings for the *programming* sense of 'Java':")
senses = {"programming": ["programming", "language", "code", "software"],
          "island": ["island", "islands", "province", "Indonesia"],
          "coffee": ["coffee", "drink", "beverage", "espresso"]}
variants = {
    "bare":        "Java is a",
    "coding ctx":  "In software engineering, Java is a",
    "island ctx":  "On the map of Indonesia, Java is a",
}
print(ph.report(variants, senses, intended="programming"))
