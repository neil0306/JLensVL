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


# --------------------------------------------------------------------------- #
# Vision-side causal swap (design doc P4)
# --------------------------------------------------------------------------- #
def _resolve_visual(model):
    for path in ("model.visual", "visual", "model.model.visual"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            if hasattr(obj, "blocks"):
                return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate .visual (ViT tower) on the model")


def batched_swap_delta(H, u_a, u_b, t, *, eps=1e-6):
    """Per-patch minimum-norm swap edit (vectorised over patches).

    For each patch ``p`` this is exactly :meth:`LensIntervention._swap_delta` in
    the 2-D span ``{u_a[p], u_b[p]}``: move the lens readings
    ``(u_a[p].h[p], u_b[p].h[p])`` a fraction ``t`` toward their swap. Patches
    where the two concept directions are ~orthogonal to ``h`` (i.e. the concept is
    not present) get a ~zero edit, so a global patch-wise swap concentrates on the
    patches that actually encode the concept.

    Args:
        H: ``[S, d]`` per-patch residuals.
        u_a, u_b: ``[S, d]`` per-patch concept lens directions.
        t: dose scalar in ``[0, 1]``.
    Returns:
        ``[S, d]`` edit (add to ``H``).
    """
    H = H.float(); u_a = u_a.float(); u_b = u_b.float()
    a0 = (u_a * H).sum(-1)                       # [S]
    b0 = (u_b * H).sum(-1)                        # [S]
    # 2x2 Gram per patch: [[aa, ab],[ab, bb]].
    aa = (u_a * u_a).sum(-1); bb = (u_b * u_b).sum(-1); ab = (u_a * u_b).sum(-1)
    aa = aa + eps; bb = bb + eps
    det = aa * bb - ab * ab
    r0 = t * (b0 - a0)                            # rhs component along u_a
    r1 = t * (a0 - b0)                            # rhs component along u_b
    # solve [[aa,ab],[ab,bb]] c = [r0,r1]
    c0 = (bb * r0 - ab * r1) / det
    c1 = (aa * r1 - ab * r0) / det
    return c0[:, None] * u_a + c1[:, None] * u_b   # [S, d]


class VisionLensIntervention:
    """Causal swap of a *visual* lens coordinate at a ViT block (P4 analogue).

    Mirrors :class:`LensIntervention` but patches inside the vision encoder. Using
    the fitted native vision lens ``Jv`` (``VisionJacobianLens``), it edits the
    block-``L`` patch residuals so the observational vision-lens reading of concept
    ``a`` and ``b`` is swapped (per patch, minimum norm), then runs the **full**
    VLM (ViT -> merger -> 32 LLM layers) and measures the dose-response on the
    model's own answer-position logits. If the model's decision follows the swap,
    the visual lens coordinate is causal -- the vision-side counterpart of the
    validated LLM-side swap.

    The concept direction ``u_c(p) = d/dh_L[p] ( sum_m nativeLens_logit_c[m] )`` is
    obtained by one backward through ``Jv``-transport + merger + tied unembed
    (i.e. the exact observational readout the native lens uses -- tied
    ``embed_tokens``, no final norm), so ``u_c`` is the local lens direction.

    Args:
        jl: a ``JLensVL`` on the *full* differentiable VLM.
        vision_lens: a fitted ``VisionJacobianLens`` (``Jv``) whose ViT tower
            matches ``jl.model``'s.
    """

    def __init__(self, jl, vision_lens):
        self.jl = jl
        self.model = jl.model
        self.lm = jl.lm
        self.vlens = vision_lens
        self.visual = _resolve_visual(self.model)
        self.merger = self.visual.merger
        # Native vision-lens unembed = tied embed_tokens, NO final norm (merger out
        # is an LLM *input* embedding, matching VisionJLens._unembed).
        self._W_embed = self.lm._embed_tokens.weight        # [vocab, d_llm]
        # Freeze params so the only autograd leaf is the captured residual (keeps
        # _concept_direction's graph minimal; JLensVL.from_pretrained only eval()s).
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ---------- native vision-lens readout (differentiable) ----------
    def _native_logits(self, H_L, block):
        """[S, d_vision] block-L residual -> [P, vocab] per merged token."""
        z = self.vlens.transport(H_L.float(), block).to(self.merger.norm.weight.dtype)
        y = self.merger(z)                                   # [P, d_llm]
        return y.float() @ self._W_embed.float().T           # [P, vocab]

    def _pick_id(self, words, H_L, block):
        ids = self.jl._word_ids(words)
        if not ids:
            raise ValueError(f"no single-token id for any of {list(words)!r}")
        with torch.no_grad():
            lg = self._native_logits(H_L, block)             # [P, vocab]
            best = max(ids, key=lambda i: float(lg[:, i].max()))
        return best

    def _concept_direction(self, tok_id, H_L, block):
        """u_c(p) = d (sum_m native_logit[m, tok_id]) / d h_L[p] -> [S, d_vision].

        Self-contained ``enable_grad`` so it works even when the caller
        (``swap``) runs under ``@torch.no_grad()``; ``model`` params are frozen in
        ``__init__`` so the only graph leaf is ``H``."""
        with torch.enable_grad():
            H = H_L.detach().float().requires_grad_(True)
            lg = self._native_logits(H, block)               # [P, vocab]
            scalar = lg[:, tok_id].sum()
            (g,) = torch.autograd.grad(scalar, H)
        return g.detach()                                    # [S, d_vision]

    # ---------- forward capture / patching ----------
    def _capture_block(self, inputs, block):
        """Run the model once (no grad); return the block-L residual [S, d_vision]
        and the clean answer-position logits."""
        store = {}
        h = self.visual.blocks[block].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", o if torch.is_tensor(o) else o[0]))
        try:
            with torch.no_grad():
                out = self.model(**inputs)
        finally:
            h.remove()
        H = store["h"]
        if H.dim() == 3:
            H = H[0]
        return H.detach(), out.logits[0, -1].float().cpu()

    @contextmanager
    def _patch_block(self, block, delta):
        """Additive hook on visual.blocks[block]: add ``delta`` [S, d_vision] to the
        (packed) block output for the duration."""
        def hook(module, inp, output):
            ten = output if torch.is_tensor(output) else output[0]
            d = delta.to(ten.dtype).to(ten.device)
            base = ten[0] if ten.dim() == 3 else ten
            edited = base + d
            edited = edited[None] if ten.dim() == 3 else edited
            if torch.is_tensor(output):
                return edited
            return (edited,) + tuple(output[1:])
        handle = self.visual.blocks[block].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def _answer_logits(self, inputs, block, delta):
        with self._patch_block(block, delta):
            out = self.model(**inputs)
        return out.logits[0, -1].float().cpu()

    # ---------- public API ----------
    def _pick_id_model(self, words, base_logits):
        """Among the single-token ids of ``words``, the one the MODEL scores highest
        at the answer position — the token the decision actually turns on."""
        ids = self.jl._word_ids(words)
        if not ids:
            raise ValueError(f"no single-token id for any of {list(words)!r}")
        return max(ids, key=lambda i: float(base_logits[i]))

    @torch.no_grad()
    def swap(self, concept_a, concept_b, *, image, question, block,
             doses=(0.0, 0.25, 0.5, 0.75, 1.0), pick_by="lens"):
        """Swap visual lens coord ``a`` <-> ``b`` at ViT ``block`` (all patches,
        per-patch min-norm) and measure the effect on the model's own answer
        logits across ``doses``. ``pick_by``: "lens" picks the concept token by the
        (in-encoder) vision-lens reading (parallels the LLM-side swap); "model"
        picks the token the model's own answer logits rank highest (the decision
        token — more faithful when the in-encoder lens is vocab-weak). Returns a
        dict with baseline scores, chosen token ids, and a dose-response."""
        if block not in self.vlens.source_blocks:
            raise ValueError(f"block {block} not in fitted Jv blocks "
                             f"{self.vlens.source_blocks}")
        inputs = self.jl._vlm_inputs(image, question)
        H_L, base_logits = self._capture_block(inputs, block)

        if pick_by == "model":
            a_id = self._pick_id_model(concept_a, base_logits)
            b_id = self._pick_id_model(concept_b, base_logits)
        else:
            a_id = self._pick_id(concept_a, H_L, block)
            b_id = self._pick_id(concept_b, H_L, block)
        u_a = self._concept_direction(a_id, H_L, block)      # [S, d_vision]
        u_b = self._concept_direction(b_id, H_L, block)

        dose = []
        for t in doses:
            if t == 0.0:
                la, lb = float(base_logits[a_id]), float(base_logits[b_id])
            else:
                delta = batched_swap_delta(H_L, u_a, u_b, float(t))
                lg = self._answer_logits(inputs, block, delta)
                la, lb = float(lg[a_id]), float(lg[b_id])
            dose.append({"t": float(t), "model_a": la, "model_b": lb, "pref": lb - la})

        base_pref = dose[0]["pref"]; full_pref = dose[-1]["pref"]
        return {
            "concept_a": self.jl.tok.decode([a_id]).strip(),
            "concept_b": self.jl.tok.decode([b_id]).strip(),
            "a_id": a_id, "b_id": b_id, "block": block,
            "baseline": {"model_a": dose[0]["model_a"], "model_b": dose[0]["model_b"]},
            "dose": dose,
            "flipped": bool((base_pref > 0) != (full_pref > 0)),
            "effect": full_pref - base_pref,
        }

    @torch.no_grad()
    def steer(self, concept_a, concept_b, *, image, question, block,
              alphas=(0.0, 0.05, 0.1, 0.2, 0.4), pick_by="model"):
        """Vision-side *steering* dose-response along the Jv concept-contrast
        direction. Unlike :meth:`swap` (a minimum-norm coordinate exchange, which is
        negligible when a concept is barely present in the ViT residual), this adds
        a controlled edit ``alpha * ||h_p|| * (u_b_hat - u_a_hat)`` per patch --
        pushing the visual lens reading toward ``b`` and away from ``a`` at a
        magnitude that is ``alpha`` of the residual norm. ``u_c_hat`` are the
        unit-normalised per-patch Jv lens directions. Reports the model's own
        answer-logit dose-response. A concept that is visually encoded moves the
        decision; a fusion-born one does not (that contrast is the finding)."""
        if block not in self.vlens.source_blocks:
            raise ValueError(f"block {block} not in fitted Jv blocks {self.vlens.source_blocks}")
        inputs = self.jl._vlm_inputs(image, question)
        H_L, base_logits = self._capture_block(inputs, block)
        if pick_by == "model":
            a_id = self._pick_id_model(concept_a, base_logits)
            b_id = self._pick_id_model(concept_b, base_logits)
        else:
            a_id = self._pick_id(concept_a, H_L, block); b_id = self._pick_id(concept_b, H_L, block)
        u_a = self._concept_direction(a_id, H_L, block)          # [S, d]
        u_b = self._concept_direction(b_id, H_L, block)
        ua = u_a / u_a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ub = u_b / u_b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        hn = H_L.float().norm(dim=-1, keepdim=True)              # [S,1]
        contrast = (ub - ua) * hn                                 # [S,d], ||.||~alpha*||h|| after scale
        dose = []
        for a in alphas:
            if a == 0.0:
                la, lb = float(base_logits[a_id]), float(base_logits[b_id])
            else:
                lg = self._answer_logits(inputs, block, float(a) * contrast)
                la, lb = float(lg[a_id]), float(lg[b_id])
            dose.append({"alpha": float(a), "model_a": la, "model_b": lb, "pref": lb - la})
        base_pref = dose[0]["pref"]; full_pref = dose[-1]["pref"]
        return {"concept_a": self.jl.tok.decode([a_id]).strip(),
                "concept_b": self.jl.tok.decode([b_id]).strip(),
                "a_id": a_id, "b_id": b_id, "block": block, "mode": "steer",
                "baseline": {"model_a": dose[0]["model_a"], "model_b": dose[0]["model_b"]},
                "dose": dose, "flipped": bool((base_pref > 0) != (full_pref > 0)),
                "effect": full_pref - base_pref}
