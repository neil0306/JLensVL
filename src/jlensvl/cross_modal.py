"""Cross-modal Jacobian ``Jx`` for JLensVL (design doc P3).

The native vision lens (`vision_lens.VisionJacobianLens`, ``Jv``) transports a ViT
block-``l`` residual to the *final ViT block* and decodes it with the model's own
merger + tied unembed -- it reads what a patch is poised to say **inside the
encoder**, before the LLM runs. This module fits the complementary observer: a
single Jacobian straight from a ViT block-``L`` patch residual, through the merger
**and all 32 LLM layers**, to the LLM final residual::

    Jx = E[ d h_final^LLM / d p_L ]            ([d_llm, d_vision] = [2560, 1024])

Read a patch end-to-end by ``unembed( Jx @ p_L )`` (the normal LLM
``final_norm + lm_head``). Comparing ``unembed(merger(Jv_L @ p))`` (in-encoder)
against ``unembed(Jx @ p)`` (post-fusion) tells you whether a concept is
**vision-native** (already legible inside the encoder) or **fusion-born** (only
emerges once the LLM fuses vision + text).

Both towers must be resident and differentiable, so this uses the *unquantized*
full VLM (e.g. ``/home/anu/qwen35_4b_dl``), not the AWQ shard. Memory: on a 24GB
card fit with ``R=1`` (no channel replication), one calibration image, a single
source block ``L``; the whole vision+LLM graph is retained across the per-channel
backward passes but recomputes nothing. Params are frozen; the autograd graph is
rooted at the captured block-``L`` output (made a ``requires_grad`` leaf).

**Deepstack caveat:** the design warns that when ``deepstack_visual_indexes`` is
non-empty some ViT-block features re-enter the LLM at deeper layers, so a pure
``Jx`` from block ``L`` understates influence. In the Qwen3.5-4B checkpoint used
here ``deepstack_visual_indexes == []`` (verified from config), so there is no
deepstack path and ``Jx`` captures the complete vision->LLM route. On a checkpoint
with deepstack this readout would be a lower bound; document per-model.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager

import torch


# --------------------------------------------------------------------------- #
# Estimator (pure torch, model-agnostic, CPU-testable)
# --------------------------------------------------------------------------- #
def cross_modal_jacobian_rows(
    source_leaf: torch.Tensor,
    target_act: torch.Tensor,
    *,
    target_positions: torch.Tensor,
    channels: torch.Tensor | None = None,
    channel_chunk: int = 1,
    retain_last: bool = False,
) -> torch.Tensor:
    """Estimate rows of the averaged cross-modal Jacobian from a live graph.

    Computes, for each requested output channel ``c``::

        Jx[c, :] = mean_{p in source} sum_{p' in target_positions}
                       d target_act[p', c] / d source_leaf[p]

    i.e. the same token-summed / source-averaged estimator as the block
    Jacobian, but *rectangular*: ``source_leaf`` and ``target_act`` may differ in
    both token count and channel width.

    Args:
        source_leaf: ``[S, Cs]`` -- the block-``L`` patch residual, a graph
            **leaf** (``requires_grad=True``); backprop lands here.
        target_act: ``[T, Ct]`` -- the LLM final residual (already indexed to the
            batch item; no batch axis). Live graph node.
        target_positions: 1-D LongTensor of rows of ``target_act`` to sum the
            cotangent over (e.g. the image-token span, or the answer position).
        channels: 1-D LongTensor of output channels to estimate (rows of ``Jx``).
            ``None`` -> all ``Ct`` channels (the full matrix).
        channel_chunk: channels per backward pass. ``1`` (default) is exact and
            cheapest in memory; a value ``k>1`` would require replicating the
            source graph ``k`` times, which this single-pass estimator does not
            do, so ``k`` must be 1 here. Kept for signature symmetry / future use.
        retain_last: keep the graph after the final channel (default False frees
            it). Set True if the caller reuses the same forward for more channels.

    Returns:
        ``Jx_rows`` ``[len(channels), Cs]`` fp32 on CPU.
    """
    if source_leaf.dim() != 2 or target_act.dim() != 2:
        raise ValueError(
            f"source_leaf [S,Cs] and target_act [T,Ct] must be 2-D; got "
            f"{tuple(source_leaf.shape)} and {tuple(target_act.shape)}"
        )
    if channel_chunk != 1:
        raise ValueError("channel_chunk>1 needs source replication; use 1 (R=1 path)")
    S, Cs = source_leaf.shape
    T, Ct = target_act.shape
    device = target_act.device
    pos = target_positions.to(device).long()
    if pos.numel() == 0:
        raise ValueError("target_positions is empty")
    if int(pos.max()) >= T or int(pos.min()) < 0:
        raise ValueError(f"target_positions out of range [0,{T})")
    if channels is None:
        channels = torch.arange(Ct)
    channels = channels.to(device).long()
    n = channels.numel()

    rows = torch.zeros(n, Cs, dtype=torch.float32)
    cot = torch.zeros_like(target_act)
    for i in range(n):
        c = int(channels[i])
        cot.zero_()
        cot[pos, c] = 1.0
        retain = retain_last or (i < n - 1)
        (grad,) = torch.autograd.grad(
            outputs=target_act, inputs=[source_leaf], grad_outputs=cot,
            retain_graph=retain,
        )
        rows[i] = grad.float().mean(dim=0).cpu()   # mean over source patches
    return rows


# --------------------------------------------------------------------------- #
# Lens container
# --------------------------------------------------------------------------- #
class CrossModalJacobianLens:
    """A fitted cross-modal Jacobian ``Jx: [d_llm, d_vision]`` + readout.

    ``Jx`` maps a ViT block-``source_block`` patch residual ``[..., d_vision]``
    into the **LLM final residual** basis via ``p @ Jx.T`` ``[..., d_llm]``; the
    caller then applies the LLM's own ``final_norm + lm_head`` (``unembed``) to
    reach vocab. Non-square (2560x1024): stored dense.

    Attributes:
        Jx: ``[d_llm, d_vision]`` fp32.
        source_block: ViT block index the transport reads *from*.
        d_llm / d_vision: output / input widths (2560 / 1024).
        readout: how the LLM target was formed ("image_span" or "answer").
        n_samples: calibration passes averaged.
        meta: provenance.
    """

    def __init__(self, Jx, *, source_block, d_llm, d_vision, readout="image_span",
                 n_samples=1, meta=None):
        self.Jx = Jx.float()
        self.source_block = int(source_block)
        self.d_llm = int(d_llm)
        self.d_vision = int(d_vision)
        self.readout = readout
        self.n_samples = int(n_samples)
        self.meta = dict(meta or {})
        if tuple(self.Jx.shape) != (self.d_llm, self.d_vision):
            raise ValueError(f"Jx shape {tuple(self.Jx.shape)} != ({d_llm},{d_vision})")

    def __repr__(self):
        return (f"CrossModalJacobianLens(source_block={self.source_block}, "
                f"d_llm={self.d_llm}, d_vision={self.d_vision}, "
                f"readout={self.readout}, n_samples={self.n_samples})")

    def save(self, path, *, dtype=torch.float16):
        torch.save({"Jx": self.Jx.to(dtype), "source_block": self.source_block,
                    "d_llm": self.d_llm, "d_vision": self.d_vision,
                    "readout": self.readout, "n_samples": self.n_samples,
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        return cls(Jx=c["Jx"], source_block=c["source_block"], d_llm=c["d_llm"],
                   d_vision=c["d_vision"], readout=c.get("readout", "image_span"),
                   n_samples=c.get("n_samples", 1), meta=c.get("meta", {}))

    def transport(self, residual, block=None):
        """Map a block-``source_block`` patch residual ``[..., d_vision]`` into the
        LLM final-residual basis ``[..., d_llm]`` (``p @ Jx.T``)."""
        if block is not None and int(block) != self.source_block:
            raise ValueError(f"lens fitted for block {self.source_block}, got {block}")
        J = self.Jx.to(device=residual.device, dtype=residual.dtype)
        return residual @ J.T


# --------------------------------------------------------------------------- #
# Fit driver (needs the full differentiable VLM)
# --------------------------------------------------------------------------- #
def _resolve_visual(model):
    """Locate the ViT tower inside a loaded HF VLM."""
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


@contextmanager
def _dual_capture(visual, llm_layers, source_block, llm_target_layer, store):
    """Root the autograd graph at ``visual.blocks[source_block]`` output (leaf) and
    capture the LLM residual at ``llm_layers[llm_target_layer]`` output.

    Stores ``store['src']`` (the block-L leaf, requires_grad) and ``store['tgt']``
    (the LLM final residual, live graph)."""
    handles = []

    def src_hook(module, inp, out):
        ten = out if torch.is_tensor(out) else out[0]
        ten.requires_grad_(True)              # make block-L output the graph leaf
        store["src"] = ten

    def tgt_hook(module, inp, out):
        store["tgt"] = out if torch.is_tensor(out) else out[0]

    handles.append(visual.blocks[source_block].register_forward_hook(src_hook))
    handles.append(llm_layers[llm_target_layer].register_forward_hook(tgt_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def fit_cross_modal_jacobian(
    jl, image, question, *, source_block, readout="image_span",
    channels=None, verbose=True,
):
    """Fit ``Jx = E[d h_final^LLM / d p_L]`` on one image (R=1 path).

    Args:
        jl: a ``JLensVL`` bound to the *full* differentiable VLM (unquantized;
            ``jl.model`` has both the ViT tower and the 32-layer LLM, ``jl.lm``
            is the LLM lens wrapper with ``.layers`` / ``.unembed``).
        image: PIL image or path (the calibration/probe image).
        question: the text prompt fed alongside the image.
        source_block: ViT block ``L`` to read from.
        readout: "image_span" (sum cotangent over the LLM image-token positions --
            the per-patch, spatially-comparable target) or "answer" (the last
            position -- the decision target).
        channels: LongTensor of LLM-final channels to estimate; None -> all d_llm
            (the full [d_llm, d_vision] matrix).
        verbose: print peak memory.

    Returns:
        ``CrossModalJacobianLens`` (single-image; average multiple with
        ``combine_cross_modal``).
    """
    model = jl.model
    lm = jl.lm
    visual = _resolve_visual(model)
    n_llm = lm.n_layers
    llm_target = n_llm - 1
    device = model.device
    # Freeze params so the retained graph's only leaf is the captured block-L
    # residual (JLensVL.from_pretrained only eval()s; live param grads would
    # needlessly retain the whole vision+LLM backward graph -> OOM on 24GB).
    for p in model.parameters():
        p.requires_grad_(False)

    inputs = jl._vlm_inputs(image, question)
    ids = inputs["input_ids"][0]
    img_tok = getattr(jl, "image_token_id", None)
    if readout == "image_span":
        if img_tok is None:
            raise ValueError("image_span readout needs model.config.image_token_id")
        positions = (ids == img_tok).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError("no image tokens found in the prompt")
    elif readout == "answer":
        positions = torch.tensor([ids.shape[0] - 1], device=ids.device)
    else:
        raise ValueError(f"unknown readout {readout!r}")

    store = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with _dual_capture(visual, lm.layers, source_block, llm_target, store), \
            torch.enable_grad():
        model(**inputs)
    src = store["src"]                                   # [S, d_vision] leaf
    tgt = store["tgt"]                                   # [1, T, d_llm] (or [T,d_llm])
    if tgt.dim() == 3:
        tgt = tgt[0]
    if src.dim() == 3:                                   # some towers keep a batch axis
        src = src[0]
    d_vision = src.shape[-1]
    d_llm = tgt.shape[-1]

    rows = cross_modal_jacobian_rows(
        src, tgt, target_positions=positions.to(device), channels=channels)
    if channels is None:
        Jx = rows                                        # [d_llm, d_vision]
    else:
        Jx = torch.zeros(d_llm, d_vision, dtype=torch.float32)
        Jx[channels.cpu().long()] = rows

    peak = (torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0)
    if verbose:
        print(f"[xfit] block L={source_block} readout={readout} "
              f"S={src.shape[0]} T={int(positions.numel())} "
              f"d_llm={d_llm} d_vision={d_vision} peak={peak:.1f}GB "
              f"channels={'all' if channels is None else int(channels.numel())}")
    del store, src, tgt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta = {"readout": readout, "peak_gb": peak,
            "n_image_tokens": int(positions.numel()),
            "partial_channels": None if channels is None else int(channels.numel())}
    return CrossModalJacobianLens(Jx, source_block=source_block, d_llm=d_llm,
                                  d_vision=d_vision, readout=readout, n_samples=1,
                                  meta=meta)


def combine_cross_modal(lenses):
    """n_samples-weighted mean of cross-modal lenses fitted on disjoint images."""
    lenses = list(lenses)
    if not lenses:
        raise ValueError("need at least one lens")
    first = lenses[0]
    if first.meta.get("partial_channels") is not None:
        raise ValueError("cannot combine partial-channel lenses (unestimated rows "
                         "are zeros); fit with channels=None to average")
    b0 = first.source_block
    for L in lenses[1:]:
        if (L.source_block != b0 or L.Jx.shape != first.Jx.shape
                or L.readout != first.readout):
            raise ValueError("lenses disagree on source_block / shape / readout")
        if L.meta.get("partial_channels") is not None:
            raise ValueError("cannot combine partial-channel lenses")
    n = sum(L.n_samples for L in lenses)
    Jx = sum(L.Jx * L.n_samples for L in lenses) / n
    return CrossModalJacobianLens(Jx, source_block=b0, d_llm=lenses[0].d_llm,
                                  d_vision=lenses[0].d_vision,
                                  readout=lenses[0].readout, n_samples=n,
                                  meta={"combined_from": len(lenses)})
