"""Fit a Jacobian lens for a model and save it.

  python scripts/fit_lens.py --model Qwen/Qwen3.5-4B --out lens.pt --n 100

Fitting does backward passes (needs a GPU with the model in it). For Qwen3.5's
Gated-DeltaNet layers, do NOT install `fla`/`causal-conv1d` so the differentiable
pure-PyTorch path is used.
"""
import argparse, os
from jlensvl import JLensVL


def load_prompts(n):
    try:
        from jlens.examples import load_wikitext_prompts
        return load_wikitext_prompts(n)
    except Exception:
        # offline fallback: a few long, generic sentences
        base = [
            "The capital of France is Paris, a city on the river Seine in western Europe.",
            "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
            "In 1969 the Apollo 11 mission landed the first two humans on the surface of the Moon.",
            "The periodic table arranges the chemical elements by increasing atomic number and shells.",
            "Isaac Newton formulated the three laws of motion and the law of universal gravitation.",
        ]
        return (base * ((n // len(base)) + 1))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="lens.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--multimodal", default="auto")
    a = ap.parse_args()

    jl = JLensVL.from_pretrained(a.model, multimodal=a.multimodal)
    prompts = load_prompts(a.n)
    print(f"fitting on {len(prompts)} prompts ...")
    jl.fit(prompts, max_seq_len=128, dim_batch=8)
    jl.save_lens(a.out)
    print("saved lens ->", a.out)


if __name__ == "__main__":
    main()
