"""BlockJacobianLens — a true Jacobian lens over a transformer's block residual stream.

A *logit lens* reads an intermediate block's residual by applying the model's own output
head directly to the raw residual ("what is this block poised to emit, decoded with the
final head"). That is the identity-transport shortcut — it silently assumes block ``l``'s
residual already lives in the final block's basis.

This module fits the missing linear transport. For each block ``l`` it estimates the
average input-output Jacobian relating that block's residual to the *final* block's
residual within one forward pass::

    J_l = E[ d x_final / d x_l ]          (averaged over tokens and calibration passes)

and the J-Lens readout transports first, then decodes with the head::

    jlens_l(x) = head( J_l @ x , e )       vs. logit-lens  head( x , e )

Estimator (`fit_block_jacobian`): for each output channel ``c`` inject a one-hot cotangent
at channel ``c`` at *every valid target token at once* and backprop to each source block's
residual. The gradient at source token ``p`` is then ``sum_{p'} d x_final[p',c] / d x_l[p]``
(the sum over target tokens); we take the mean over source tokens ``p``. That yields row
``c`` of the ``[C, C]`` matrix ``J_l``. To compute ``R`` channels per backward pass we
replicate the forward ``R`` times along the batch axis (batch element ``b`` carries the
one-hot for channel ``dim_start + b``), so ``channel_batch == batch_size``.

Storage/readout API (``save`` -> a ``.pt`` dict with fp16 ``J`` matrices, ``load``,
``merge``, ``transport``) so a fitted lens is handled uniformly downstream.

Pure-torch, no external engine import: the module only needs the *live-graph* per-block
residual tensors and does the autograd bookkeeping. The model-specific capture of those
residuals lives in the caller (e.g. `vision_lens.VisionJLens`).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch


def fit_block_jacobian(
    source_acts: dict[int, torch.Tensor],
    target_act: torch.Tensor,
    *,
    valid_tokens: torch.Tensor | None = None,
    skip_first: int = 0,
    token_stride: int = 1,
) -> dict[int, torch.Tensor]:
    """Estimate one pass's per-block Jacobian rows from live-graph residuals.

    Args:
        source_acts: ``{block_idx: Tensor[R, L, C]}`` -- each source block's residual
            output, a live (graph-retained, requires_grad) tensor. ``R`` is the batch
            replication factor: batch element ``b`` will receive the one-hot cotangent for
            output channel ``dim_start + b``, so ``R`` channels are done per backward pass.
            All ``R`` rows are identical replicas of the same input.
        target_act: ``Tensor[R, L, C]`` -- the final (target) block's residual output,
            same live graph.
        valid_tokens: explicit 1-D LongTensor of target/source token indices to use. If
            None, uses ``range(L)`` after applying ``skip_first`` and ``token_stride``.
        skip_first: drop this many leading tokens from the average (analog of an LM
            estimator's attention-sink skip; default 0 -- use it only if the model has a
            clear sink token).
        token_stride: subsample tokens by this stride for the average (memory/speed knob;
            the cotangent still lands on the subsampled valid tokens only).

    Returns:
        ``{block_idx: Tensor[C, C] fp32 on CPU}`` -- for each source block, row ``c`` is
        ``mean_{p in valid} sum_{p' in valid} d target[p',c] / d source[p]``.
    """
    if target_act.dim() != 3:
        raise ValueError(f"target_act must be [R, L, C], got {tuple(target_act.shape)}")
    R, L, C = target_act.shape
    device = target_act.device
    layers = sorted(source_acts)
    for l in layers:
        if source_acts[l].shape != target_act.shape:
            raise ValueError(
                f"source block {l} shape {tuple(source_acts[l].shape)} != "
                f"target {tuple(target_act.shape)}"
            )

    if valid_tokens is None:
        valid = torch.arange(skip_first, L, token_stride, device=device)
    else:
        valid = valid_tokens.to(device)
    if valid.numel() == 0:
        raise ValueError("no valid tokens to average over")

    J = {l: torch.zeros(C, C, dtype=torch.float32) for l in layers}
    cot = torch.zeros_like(target_act)
    batch_idx = torch.arange(R, device=device)
    n_passes = math.ceil(C / R)

    for pass_idx, dim_start in enumerate(range(0, C, R)):
        n_dims = min(R, C - dim_start)
        # One-hot cotangent: batch element b -> channel (dim_start + b), at every valid
        # target token. Yields rows dim_start .. dim_start+n_dims of J_l.
        cot.zero_()
        cot[
            batch_idx[:n_dims, None],
            valid[None, :],
            dim_start + batch_idx[:n_dims, None],
        ] = 1.0
        grads = torch.autograd.grad(
            outputs=target_act,
            inputs=[source_acts[l] for l in layers],
            grad_outputs=cot,
            retain_graph=(pass_idx < n_passes - 1),
        )
        for l, g in zip(layers, grads):
            # g: [R, L, C]; take the n_dims live batch elements, mean over valid source
            # tokens -> n_dims rows of J_l.
            rows = g[:n_dims][:, valid, :].float().mean(dim=1)  # [n_dims, C]
            J[l][dim_start : dim_start + n_dims, :] = rows.cpu()
        del grads

    return J


class BlockJacobianLens:
    """A fitted block Jacobian lens: per-block ``J_l`` matrices + the transport readout.

    Each ``J_l`` maps a block ``l`` residual ``[..., C]`` into the final (target) block's
    residual basis via ``h @ J_l.T``. The caller then applies the model's own head to
    decode it.

    Attributes:
        jacobians: ``{block_idx: Tensor[C, C]}``.
        source_blocks: sorted list of fitted block indices.
        n_samples: number of calibration passes (prompt x call) averaged.
        d_model: residual width ``C``.
        target_block: the block index the transport maps *to* (the final block by default).
        meta: free-form dict of provenance.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_samples: int,
        d_model: int,
        target_block: int,
        meta: dict | None = None,
    ) -> None:
        self.jacobians = {int(l): J.float() for l, J in jacobians.items()}
        self.source_blocks = sorted(self.jacobians)
        self.n_samples = n_samples
        self.d_model = d_model
        self.target_block = target_block
        self.meta = dict(meta or {})

    def __repr__(self) -> str:
        return (
            f"BlockJacobianLens(d_model={self.d_model}, n_samples={self.n_samples}, "
            f"target_block={self.target_block}, source_blocks="
            f"[{self.source_blocks[0]}..{self.source_blocks[-1]}] "
            f"({len(self.source_blocks)} blocks))"
        )

    # ---------- storage ----------
    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save to ``path`` as a ``.pt`` dict; ``J`` stored as ``dtype`` (default fp16,
        halving file size -- entries are O(1) so fp16 range is not a constraint)."""
        torch.save(
            {
                "J": {l: J.to(dtype) for l, J in self.jacobians.items()},
                "n_samples": self.n_samples,
                "source_blocks": self.source_blocks,
                "d_model": self.d_model,
                "target_block": self.target_block,
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "BlockJacobianLens":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "J" not in ckpt:
            raise ValueError(f"{path} is not a BlockJacobianLens file (keys {sorted(ckpt)})")
        return cls(
            jacobians=ckpt["J"],
            n_samples=ckpt["n_samples"],
            d_model=ckpt["d_model"],
            target_block=ckpt["target_block"],
            meta=ckpt.get("meta", {}),
        )

    @classmethod
    def merge(cls, lenses: Sequence["BlockJacobianLens"]) -> "BlockJacobianLens":
        """Combine lenses fitted on disjoint calibration subsets (n_samples-weighted mean)."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.source_blocks != first.source_blocks or other.d_model != first.d_model:
                raise ValueError("lenses disagree on source_blocks / d_model")
            if other.target_block != first.target_block:
                raise ValueError(
                    f"lenses disagree on target_block "
                    f"({first.target_block} vs {other.target_block}); "
                    f"transports into different blocks cannot be merged")
        n_total = sum(l.n_samples for l in lenses)
        merged = {}
        for b in first.source_blocks:
            merged[b] = sum(l.jacobians[b] * l.n_samples for l in lenses) / n_total
        return cls(
            jacobians=merged,
            n_samples=n_total,
            d_model=first.d_model,
            target_block=first.target_block,
            meta={"merged_from": len(lenses)},
        )

    # ---------- readout ----------
    def transport(self, residual: torch.Tensor, block: int) -> torch.Tensor:
        """Map a block-``l`` residual into the final block's basis: ``h @ J_l.T``.

        Args:
            residual: ``[..., C]``.
            block: source block index (must be in :attr:`source_blocks`).
        """
        J = self.jacobians[block].to(device=residual.device, dtype=residual.dtype)
        return residual @ J.T


class RunningJacobianAccumulator:
    """Streams per-pass Jacobians from :func:`fit_block_jacobian` into a running mean.

    Kept separate from the lens object so a driver can accumulate over a handful of
    calibration passes (freeing each pass's autograd graph before the next) and only
    materialise the final :class:`BlockJacobianLens` at the end.
    """

    def __init__(self, source_blocks: Sequence[int], d_model: int, target_block: int):
        self.source_blocks = sorted(int(b) for b in source_blocks)
        self.d_model = d_model
        self.target_block = target_block
        self._sum = {b: torch.zeros(d_model, d_model, dtype=torch.float32) for b in self.source_blocks}
        self.n_samples = 0
        self.per_sample_norm: list[float] = []

    def add(self, per_pass_J: dict[int, torch.Tensor]) -> float:
        """Add one pass's Jacobians. Returns this pass's max ||J_l||/sqrt(d) (outlier flag)."""
        sqrt_d = math.sqrt(self.d_model)
        norm = 0.0
        for b in self.source_blocks:
            if b not in per_pass_J:
                raise KeyError(f"pass Jacobian missing block {b}")
            self._sum[b] += per_pass_J[b]
            norm = max(norm, per_pass_J[b].norm().item() / sqrt_d)
        self.n_samples += 1
        self.per_sample_norm.append(norm)
        return norm

    def finalize(self, meta: dict | None = None) -> BlockJacobianLens:
        if self.n_samples == 0:
            raise RuntimeError("no calibration passes were accumulated")
        mean = {b: self._sum[b] / self.n_samples for b in self.source_blocks}
        m = {"per_sample_max_norm": self.per_sample_norm}
        m.update(meta or {})
        return BlockJacobianLens(
            jacobians=mean,
            n_samples=self.n_samples,
            d_model=self.d_model,
            target_block=self.target_block,
            meta=m,
        )
