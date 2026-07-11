"""Causal lens-coordinate interventions for JLensVL.

The J-Lens readout is *observational*: it tells you what concept a residual is
poised to decode into, but not whether that concept is causally load-bearing for
the model's final decision. This module adds the causal test.

Idea (lens-coordinate swap patching): the lens score of concept ``c`` at layer
``L`` is, to first order (ignoring the final norm's nonlinearity),

    score_c(h) = unembed(J_L h)[c] ~= W_U[c] . (J_L h) = (W_U[c] @ J_L) . h = u_c . h

so ``u_c = W_U[c] @ J_L`` is the *lens direction* of concept ``c`` in the
residual stream at layer ``L`` (the gradient of the lens logit wrt the
residual). To causally interrogate the lens we edit the residual at layer ``L``,
position ``p`` so that the lens *readings* along two concepts ``a`` and ``b`` are
swapped, then let the rest of the network run and see whether the model's real
final decision follows. If it does, the lens coordinate is causal, not just
correlational.

The swap is the minimum-norm edit inside ``span{u_a, u_b}`` that maps
``(u_a.h, u_b.h)`` -> ``(u_b.h, u_a.h)``. A dose parameter ``t in [0,1]``
interpolates (``t=0`` no-op, ``t=1`` full swap), giving a dose-response curve.

Everything here is forward-only at inference: the ``J_L`` come from the fitted
lens; each measurement is one forward pass with a single additive hook.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch


class LensIntervention:
    """Causal lens-coordinate swap patching bound to a fitted ``JLensVL``.

    Args:
        jl: a ``JLensVL`` whose ``.lens`` is fitted.
    """

    def __init__(self, jl):
        if getattr(jl, "lens", None) is None:
            raise RuntimeError("LensIntervention needs a fitted lens (jl.lens is None)")
        self.jl = jl
        self.lm = jl.lm
        self.model = jl.model
        self.lens = jl.lens
        # Unembedding matrix W_U: [vocab, d_model].
        self._W_U = self.lm._lm_head.weight

    # ---------- inputs ----------
    def _build_inputs(self, *, prompt=None, image=None, question=None):
        """Return HF model inputs on the model device for a text or VLM query."""
        if image is not None:
            if question is None:
                raise ValueError("pass question= alongside image=")
            return self.jl._vlm_inputs(image, question)
        if prompt is None:
            raise ValueError("pass either prompt= (text) or image=+question= (VLM)")
        ids = self.lm.encode(prompt)
        return {"input_ids": ids}

    def _resolve_position(self, inputs, position):
        seq_len = inputs["input_ids"].shape[1]
        p = int(position)
        return p if p >= 0 else seq_len + p

    # ---------- concept directions ----------
    def _pick_id(self, words, residual, layer):
        """Among the single-token ids of ``words``, the one the lens scores
        highest at (layer, this residual) — matches ``concept_race``'s max."""
        ids = self.jl._word_ids(words)
        if not ids:
            raise ValueError(f"no single-token id for any of {list(words)!r}")
        z = self.lens.transport(residual.float(), layer)
        logits = self.lm.unembed(z[None].to(residual.device))[0]
        best = max(ids, key=lambda i: float(logits[i]))
        return best

    def _direction(self, tok_id, layer):
        """Lens direction u_c = W_U[c] @ J_L in residual space (float, cpu)."""
        w = self._W_U[tok_id].detach().float().cpu()          # [d_model]
        J = self.lens.jacobians[layer].float().cpu()          # [d_model, d_model]
        return w @ J                                          # [d_model]

    # ---------- the swap edit ----------
    @staticmethod
    def _swap_delta(h, u_a, u_b, t):
        """Minimum-norm residual edit in span{u_a,u_b} that moves the lens
        readings (u_a.h, u_b.h) a fraction ``t`` of the way toward their swap.

        Solve for c in R^2 with U = [u_a u_b] (d x 2):
            U^T (h + U c) = s + t * [b0 - a0, a0 - b0]
          =>  (U^T U) c = t * [b0 - a0, a0 - b0]
        delta = U c.  ``t=1`` is a full swap, ``t=0`` a no-op.
        """
        h = h.float()
        U = torch.stack([u_a, u_b], dim=1)                    # [d, 2]
        a0 = float(u_a @ h)
        b0 = float(u_b @ h)
        rhs = t * torch.tensor([b0 - a0, a0 - b0], dtype=torch.float32)
        G = U.T @ U                                           # [2, 2]
        # Tikhonov guard for near-parallel directions.
        G = G + 1e-6 * torch.eye(2)
        c = torch.linalg.solve(G, rhs)                        # [2]
        return U @ c                                          # [d]

    # ---------- patching forward pass ----------
    @contextmanager
    def _patch(self, layer, delta, position):
        """Install an additive forward hook on block ``layer`` that adds
        ``delta`` to the residual at ``position`` for the duration."""
        block = self.lm.layers[layer]

        def hook(module, inp, output):
            ten = output if torch.is_tensor(output) else output[0]
            d = delta.to(ten.dtype).to(ten.device)
            ten = ten.clone()
            ten[:, position, :] = ten[:, position, :] + d
            if torch.is_tensor(output):
                return ten
            return (ten,) + tuple(output[1:])

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def _forward_logits(self, inputs, layer, position, delta=None):
        """One forward pass; returns (final_logits_at_pos, residual_at_layer)."""
        from jlens.hooks import ActivationRecorder
        ctx = self._patch(layer, delta, position) if delta is not None else _null()
        with ActivationRecorder(self.lm.layers, at=[layer]) as rec, ctx:
            out = self.model(**inputs)
            resid = rec.activations[layer][0, position].detach().float().cpu()
        logits = out.logits[0, position].float().cpu()
        return logits, resid

    # ---------- public API ----------
    @torch.no_grad()
    def swap(self, concept_a, concept_b, *, prompt=None, image=None, question=None,
             layer=None, position=-1, doses=(0.0, 0.25, 0.5, 0.75, 1.0)):
        """Run a lens-coordinate swap of ``concept_a`` <-> ``concept_b`` at
        ``layer``/``position`` and measure the causal effect on the model's own
        final logits, across ``doses``.

        ``concept_a``/``concept_b`` are word lists (like ``concept_race``).
        Returns a dict with baseline lens/model scores, the chosen token ids, and
        a dose-response list ``[{t, model_a, model_b, pref}]`` (``pref`` =
        model_b - model_a; a sign flip vs. baseline means the swap is causal).
        """
        inputs = self._build_inputs(prompt=prompt, image=image, question=question)
        p = self._resolve_position(inputs, position)
        layer = int(layer if layer is not None else self.lens.source_layers[len(self.lens.source_layers) // 2])
        if layer not in self.lens.source_layers:
            raise ValueError(f"layer {layer} not in fitted source_layers {self.lens.source_layers}")

        # Baseline: clean residual at (layer, p) + model logits.
        base_logits, h = self._forward_logits(inputs, layer, p, delta=None)
        a_id = self._pick_id(concept_a, h, layer)
        b_id = self._pick_id(concept_b, h, layer)
        u_a = self._direction(a_id, layer)
        u_b = self._direction(b_id, layer)

        # Lens readout at baseline (what the observational lens says).
        z = self.lens.transport(h, layer)
        lens_logits = self.lm.unembed(z[None])[0].float().cpu()

        dose = []
        for t in doses:
            if t == 0.0:
                la, lb = float(base_logits[a_id]), float(base_logits[b_id])
            else:
                delta = self._swap_delta(h, u_a, u_b, float(t))
                lg, _ = self._forward_logits(inputs, layer, p, delta=delta)
                la, lb = float(lg[a_id]), float(lg[b_id])
            dose.append({"t": float(t), "model_a": la, "model_b": lb, "pref": lb - la})

        base_pref = dose[0]["pref"]
        full_pref = dose[-1]["pref"]
        flipped = (base_pref > 0) != (full_pref > 0)
        return {
            "concept_a": self.jl.tok.decode([a_id]).strip(),
            "concept_b": self.jl.tok.decode([b_id]).strip(),
            "a_id": a_id, "b_id": b_id, "layer": layer, "position": p,
            "baseline": {
                "lens_a": float(lens_logits[a_id]), "lens_b": float(lens_logits[b_id]),
                "model_a": dose[0]["model_a"], "model_b": dose[0]["model_b"],
            },
            "dose": dose,
            "flipped": bool(flipped),
            "effect": full_pref - base_pref,
        }


@contextmanager
def _null():
    yield
