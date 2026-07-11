"""Fit a Jacobian lens for a model and save it.

  python scripts/fit_lens.py --model Qwen/Qwen3.5-4B --out lens.pt --n 100

Fitting does backward passes (needs a GPU with the model in it). For Qwen3.5's
Gated-DeltaNet layers, do NOT install `fla`/`causal-conv1d` so the differentiable
pure-PyTorch path is used.

On Apple Silicon (device="auto" picks MPS when no CUDA is present) a 4B model
fits at roughly 20-25 min/prompt, so n=100 can run the better part of two days.
--checkpoint makes that safe to interrupt/resume; progress logging (on by
default) prints one line per completed prompt so a long run isn't silent.
"""
import argparse, os
import jlens
from jlensvl import JLensVL

jlens.configure_logging()


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
    ap.add_argument("--device", default="auto", help='"auto" (default), "cuda", "mps", or "cpu"')
    ap.add_argument("--dim-batch", type=int, default=8,
                     help="output dims computed per backward pass; higher uses more GPU memory "
                          "but total backward FLOPs are unchanged, so this mainly trades memory "
                          "for fewer/bigger kernel launches, not raw speed")
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--checkpoint", default=None,
                     help="resumable checkpoint path (defaults to <out>.ckpt); "
                          "re-running the same command after an interruption resumes from it")
    ap.add_argument("--no-resume", action="store_true",
                     help="ignore an existing checkpoint and start fresh")
    a = ap.parse_args()
    checkpoint = a.checkpoint or (a.out + ".ckpt")

    jl = JLensVL.from_pretrained(a.model, multimodal=a.multimodal, device=a.device)
    prompts = load_prompts(a.n)
    print(f"fitting on {len(prompts)} prompts ... (dim_batch={a.dim_batch}, "
          f"max_seq_len={a.max_seq_len}, checkpoint={checkpoint})")
    jl.fit(prompts, max_seq_len=a.max_seq_len, dim_batch=a.dim_batch,
           checkpoint_path=checkpoint, checkpoint_every=1, resume=not a.no_resume)
    jl.save_lens(a.out)
    print("saved lens ->", a.out)


if __name__ == "__main__":
    main()
