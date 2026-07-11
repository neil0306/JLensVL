"""Causal lens-coordinate swap — is the J-Lens readout load-bearing?

Observational lenses show a concept *is present*; this shows it *matters*. We
swap two concepts' lens coordinates mid-stack and watch the model's own final
logits follow. Text demo (no image needed):

    python examples/09_intervention.py --lens lens_qwen35_4b_final.pt

Needs the model resident on GPU (set CUDA_VISIBLE_DEVICES to a free GPU).
"""
import argparse

from jlensvl import JLensVL
from jlensvl.interventions import LensIntervention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--prompt",
                    default="Fact: The capital city of France is")
    ap.add_argument("--a", nargs="+", default=["Paris"])
    ap.add_argument("--b", nargs="+", default=["Rome"])
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()

    jl = JLensVL.from_pretrained(a.model, lens=a.lens, device="cuda")
    iv = LensIntervention(jl)
    r = iv.swap(a.a, a.b, prompt=a.prompt, layer=a.layer)

    print(f"\nprompt: {a.prompt!r}")
    print(f"swap  : {r['concept_a']!r} <-> {r['concept_b']!r}  "
          f"at layer {r['layer']} pos {r['position']}")
    b = r["baseline"]
    print(f"lens (observational): {r['concept_a']}={b['lens_a']:.2f}  "
          f"{r['concept_b']}={b['lens_b']:.2f}")
    print("dose-response (model's OWN final logits):")
    print(f"   t     {r['concept_a']:>8}  {r['concept_b']:>8}   pref(b-a)")
    for d in r["dose"]:
        print(f"  {d['t']:.2f}  {d['model_a']:8.2f}  {d['model_b']:8.2f}   {d['pref']:+.2f}")
    verdict = "CAUSAL (decision flipped)" if r["flipped"] else "not decisive"
    print(f"=> {verdict};  net effect on pref = {r['effect']:+.2f}")


if __name__ == "__main__":
    main()
