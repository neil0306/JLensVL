"""Forward-only prompt diagnostics via the J-Lens.

Idea: the J-Lens reads what a model is *poised to say* at the end of a prompt,
before it generates. So you can (a) see the concept a prompt steers toward and
how decisive it is, and (b) rank alternative phrasings by how strongly they push
the model to an *intended* sense versus competing senses.

This is a diagnostic/comparison aid (white-box; needs the model + a fitted lens),
not an automatic prompt generator. Every call is a single forward pass.
"""
from __future__ import annotations


class PromptHelper:
    def __init__(self, jl):
        """jl: a fitted JLensVL instance."""
        self.jl = jl

    def _layer(self, layer):
        return layer if layer is not None else self.jl.lens.source_layers[-2]

    def poised(self, prompt, *, layer=None, k=6):
        """What is the model poised to say at the end of `prompt`?
        Returns top-k tokens + a decisiveness margin (top1 - top2)."""
        jl = self.jl; jl._require_lens(); L = self._layer(layer)
        ll, _, _ = jl.lens.apply(jl.lm, prompt, positions=[-1], layers=[L], use_jacobian=True)
        vals, idx = ll[L][0].topk(k)
        toks = jl._decode(idx.tolist())
        return {"tokens": toks, "top1": toks[0], "margin": float(vals[0] - vals[1]), "layer": L}

    def sense_scores(self, prompt, senses, *, layer=None):
        """J-Lens score of each candidate sense at the answer position.
        `senses` = {name: [words]}. Higher = the model leans that way."""
        jl = self.jl; jl._require_lens(); L = self._layer(layer)
        ll, _, _ = jl.lens.apply(jl.lm, prompt, positions=[-1], layers=[L], use_jacobian=True)
        logits = ll[L][0]
        return {n: float(logits[jl._word_ids(w)].max()) for n, w in senses.items()}

    def rank_prompts(self, variants, senses, intended, *, layer=None):
        """Rank prompt `variants` by how well they steer to the `intended` sense
        versus the strongest competing sense.

        Args:
            variants: list of prompts, or {name: prompt}.
            senses: {sense_name: [words]} candidate meanings.
            intended: the sense_name you want the prompt to elicit.
        Returns: list of dicts sorted best-first by (intended - best_competitor).
        """
        items = variants.items() if isinstance(variants, dict) else [(p, p) for p in variants]
        rows = []
        for name, prompt in items:
            sc = self.sense_scores(prompt, senses, layer=layer)
            comp = max((v for k, v in sc.items() if k != intended), default=float("-inf"))
            rows.append({"name": name, "prompt": prompt, "scores": sc,
                         "intended": sc.get(intended), "best_competitor": comp,
                         "margin": sc.get(intended, float("-inf")) - comp})
        rows.sort(key=lambda r: r["margin"], reverse=True)
        return rows

    def report(self, variants, senses, intended, *, layer=None, width=22):
        """Human-readable, visual ranking of prompt variants: ASCII bars of each
        sense's J-Lens score, a per-prompt verdict, and the overall winner.
        Returns a printable string."""
        rows = self.rank_prompts(variants, senses, intended, layer=layer)
        allv = [v for r in rows for v in r["scores"].values() if v is not None]
        mx = max(allv) if allv else 1.0
        out = [f"Prompt-Helper report — intended sense: {intended!r}  "
               f"(J-Lens @ layer {self._layer(layer)})", "=" * 68]
        for rank, r in enumerate(rows, 1):
            m = r["margin"]
            verdict = "✓ CLEAR" if m >= 3 else ("~ weak/ambiguous" if m > 0 else "✗ OFF-TARGET")
            out.append(f"\n#{rank}  [{r['name']}]  {r['prompt']!r}")
            for s, v in sorted(r["scores"].items(), key=lambda x: -(x[1] or -1e9)):
                nb = int(round(width * max(v or 0, 0) / mx)) if mx > 0 else 0
                tag = "  ← intended" if s == intended else ""
                out.append(f"    {s:12s} |{'█' * nb:<{width}}| {v:6.2f}{tag}")
            out.append(f"    → margin (intended − best competitor) = {m:+.2f}   [{verdict}]")
        out.append(f"\nRANKING (best first): {' > '.join(r['name'] for r in rows)}")
        b = rows[0]
        out.append(f"VERDICT: use [{b['name']}] — it steers the model to {intended!r} "
                   f"with a {b['margin']:+.2f} margin over the next sense.")
        return "\n".join(out)
