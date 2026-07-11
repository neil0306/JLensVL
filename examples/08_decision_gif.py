"""One inference -> a layer-by-layer GIF of how the candidate answers form.

The candidate answers are AUTO-DETECTED from the model's own top-k next tokens at
the decision point (no hardcoding). Great for "detect X?" VLM tasks: watch the
verdict classes compete across layers, like the boot->Italy->euro example.

  MODEL_DIR=... LENS=lens.pt IMG=frame.png SYS="<system prompt>" \
  Q="<user instruction>" PREFILL='{"tier": "' python examples/08_decision_gif.py
"""
import os
from jlensvl import JLensVL, viz

jl = JLensVL.from_pretrained(os.environ.get("MODEL_DIR", "Qwen/Qwen3.5-4B"),
                             lens=os.environ.get("LENS", "lens.pt"))

trace = jl.decision_trace(
    os.environ.get("IMG", "images/dog.jpg"),
    os.environ.get("Q", "What animal is this? Answer with one word."),
    system=os.environ.get("SYS") or None,
    prefill=os.environ.get("PREFILL", ""),   # e.g. '{"tier": "' for JSON-verdict prompts
    concepts=None,                            # None -> auto-detect the model's own candidates
)
print("auto-detected candidates:", trace["candidates"])
viz.decision_gif(trace, out_gif="decision.gif", title="J-Lens decision")
print("wrote decision.gif")
