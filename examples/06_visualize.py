"""Generate self-contained HTML visualizations: the layer x position slice grid
(text and VLM) and a concept-race chart. Open the .html files in any browser."""
import os
from jlensvl import JLensVL, viz

MID = os.environ.get("MODEL_DIR", "Qwen/Qwen3.5-4B")
LENS = os.environ.get("LENS", "lens_qwen35_4b_final.pt")
DEV = os.environ.get("DEV", "cuda:0")

jl = JLensVL.from_pretrained(MID, lens=LENS, device=DEV)

# 1) text slice grid — watch "euro"/"Italian" climb the layers
viz.slice_grid_html(
    jl, "Fact: The currency used in the country shaped like a boot is the",
    layers=list(range(16, 31)), topk=5, out_path="slice_text.html",
    title="J-Lens slice — boot → Italy → euro")
print("wrote slice_text.html")

# 2) VLM slice grid + concept race (needs an image)
img = "images/dog.jpg"
if os.path.exists(img):
    viz.slice_grid_image_html(
        jl, img, "What animal is in this image? Answer with one word.",
        layers=list(range(16, 31)), out_path="slice_vlm.html",
        title="J-Lens VLM slice — dog photo")
    print("wrote slice_vlm.html")
    race = jl.concept_race(img, "This is a cat. What animal is in this image? Answer with one word.",
                           {"dog": ["dog", "puppy", "pug"], "cat": ["cat", "kitten"]},
                           layers=list(range(12, 31)))
    viz.race_chart_html(race, "dog", "cat", out_path="race.html",
                        title="concept race — dog photo, text says 'cat'")
    print("wrote race.html")
