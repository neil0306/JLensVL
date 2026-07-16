"""Shared helpers for the JLensVL examples — keep them path-free and reproducible.

Nothing here is hardcoded to one machine: the base model defaults to the public
``Qwen/Qwen3.5-4B`` checkpoint, and the two fitted lenses are pulled from the public
HF repo ``neil0306/JLensVL-lenses`` on first use (cached by huggingface_hub). Every
default is overridable by an env var so you can point at local copies.
"""
import os

MODEL_ID = os.environ.get("JLENSVL_MODEL", "Qwen/Qwen3.5-4B")
LENS_REPO = os.environ.get("JLENSVL_LENS_REPO", "neil0306/JLensVL-lenses")

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_IMAGES = os.path.join(HERE, "assets", "vision")     # 7 object photos shipped in-repo


def lens_path(kind):
    """Resolve a fitted-lens file, downloading it from HF on first use.

    kind='vision' -> vision_jacobian_lens.pt ; kind='llm' -> lens_qwen35_4b_final.pt.
    Override with JLENSVL_VISION_LENS / JLENSVL_LLM_LENS to use a local file instead.
    """
    fname = {"vision": "vision_jacobian_lens.pt", "llm": "lens_qwen35_4b_final.pt"}[kind]
    env = {"vision": "JLENSVL_VISION_LENS", "llm": "JLENSVL_LLM_LENS"}[kind]
    override = os.environ.get(env)
    if override:
        return override
    from huggingface_hub import hf_hub_download
    return hf_hub_download(LENS_REPO, fname)
