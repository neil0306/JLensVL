"""VisionJLens -- a Jacobian-Lens observer for the *vision tower* of a Qwen3.5 VLM.

The package's existing `JLensVL` (core.py) lenses the LLM decoder's residual stream at
image-token positions. It never looks inside the vision encoder itself. This module does:
it reads what each **ViT block** of `model.model.visual` is *poised to say*, decoded into
the **language vocabulary**, at every merged patch position -> a spatial [14x14 x vocab] map.

Two readout modes, both mapping a vision block-l residual into the LLM vocab. The model's
own "head" -- the patch **merger** (LayerNorm -> 2x2 spatial concat -> fc1 -> GELU -> fc2 ->
2560-d) followed by the tied **unembed** (lm_head = embed_tokens.T) -- is what turns a
final-block residual into vocab logits. A lens differs only in *how it gets a block-l
residual into the final block's basis before that head is applied*:

* **Naive vision-logit-lens** (baseline, reproduces Neo et al. 2024 "logit lens on image
  tokens"): apply the head directly to the raw block-l residual, skipping blocks l+1..23::

      naive_l(h) = unembed( merger( h_l ) )

  the identity-transport shortcut -- it assumes block l already lives in the final basis. It
  is noisy mid-stack; and on Qwen the deep merged tokens are dominated by global/register
  signal (uniform-background patches out-score the object).

* **Fitted vision-Jacobian-lens** (our improvement, a direct mirror of the language J-Lens):
  fit the missing linear transport from block l's residual to the **final block's** residual,
  then apply the model's own head::

      J_l = E[ d h_23 / d h_l ]           (both in the 1024-d pre-merge residual space)
      jlens_l(h) = unembed( merger( J_l @ h_l ) )

  ``J_l`` is a ``[1024, 1024]`` matrix, estimated by backprop and averaged over patch tokens
  and calibration images -- the same estimator as `block_jacobian.fit_block_jacobian`
  (reused here). Row ``c`` of ``J_l`` is ``mean_p sum_{p'} d h_23[p',c] / d h_l[p]``.
  Transport happens at pre-merge (per-patch) granularity; the merger's own 2x2 reshape then
  restores the merged-grid [14x14] layout at readout, so spatial correspondence is preserved.

  At the last block (l=23) the transport is the identity (``J_23 = I``), so ``jlens_23`` is
  *exactly* ``naive_23`` -- a built-in sanity anchor (verified: cos=1.0, top-1 agree=1.0).

Weights: the LLM decoder is not needed at all. We load only the (unquantized bf16) vision
tower + merger + the tied unembedding (embed_tokens). See `VisionJLens.from_qwen35`.
"""
from __future__ import annotations

import math
import os

import torch

# Generic per-block Jacobian running-mean accumulator (pure-torch, no external engine).
try:
    from .block_jacobian import RunningJacobianAccumulator
except ImportError:                                   # allow load-by-path (see examples/)
    import importlib.util
    _p = os.path.join(os.path.dirname(__file__), "block_jacobian.py")
    _s = importlib.util.spec_from_file_location("jlensvl_block_jacobian", _p)
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    RunningJacobianAccumulator = _m.RunningJacobianAccumulator


def fit_block_jacobian_packed(source_acts, target_act, *, R):
    """`block_jacobian.fit_block_jacobian`, for a *packed* vision forward.

    Same one-hot-cotangent estimator (row ``c`` = ``mean_p sum_{p'} d target[p',c]/d src[p]``,
    ``R`` output channels per backward pass), but the Qwen vision tower packs its ``R``
    replicated images as ``[R*S, C]`` with no batch axis (attention is kept intra-image by
    ``cu_seqlens``). We therefore differentiate w.r.t. the packed graph node directly and do
    the per-replica bookkeeping here: replica ``b`` carries the one-hot for output channel
    ``dim_start+b`` at *its own* ``S`` tokens.

    Args:
        source_acts: ``{block: Tensor[R*S, C]}`` live packed residuals (require grad).
        target_act: ``Tensor[R*S, C]`` live packed final-block residual.
        R: replication factor (== channels done per backward pass).

    Returns:
        ``{block: Tensor[C, C] fp32 CPU}``.
    """
    layers = sorted(source_acts)
    RS, C = target_act.shape
    if RS % R:
        raise ValueError(f"R={R} does not divide packed rows {RS}")
    S = RS // R
    device = target_act.device
    J = {l: torch.zeros(C, C, dtype=torch.float32) for l in layers}
    cot = torch.zeros_like(target_act)
    b_idx = torch.arange(R, device=device)
    row_base = (b_idx * S)[:, None] + torch.arange(S, device=device)[None, :]   # [R,S]
    n_passes = math.ceil(C / R)
    for pass_idx, dim_start in enumerate(range(0, C, R)):
        n = min(R, C - dim_start)
        cot.zero_()
        cot[row_base[:n].reshape(-1), (dim_start + b_idx[:n]).repeat_interleave(S)] = 1.0
        grads = torch.autograd.grad(
            outputs=target_act, inputs=[source_acts[l] for l in layers],
            grad_outputs=cot, retain_graph=(pass_idx < n_passes - 1))
        for l, g in zip(layers, grads):
            rows = g.view(R, S, C)[:n].float().mean(dim=1)      # [n, C]
            J[l][dim_start:dim_start + n, :] = rows.cpu()
        del grads
    return J


class VisionJacobianLens:
    """Fitted per-block vision Jacobians ``J_l: [1024, 1024]`` + the transport readout.

    Each ``J_l`` maps a ViT block's pre-merge residual ``[..., 1024]`` into the final block's
    residual basis via ``h @ J_l.T``; the caller then applies the model's own merger + tied
    unembed to reach vocab.
    """

    def __init__(self, jacobians, *, n_samples, d_model, target_block, meta=None):
        self.jacobians = {int(l): J.float() for l, J in jacobians.items()}
        self.source_blocks = sorted(self.jacobians)
        self.n_samples = n_samples
        self.d_model = d_model          # 1024 (vision hidden dim)
        self.target_block = target_block
        self.meta = dict(meta or {})

    def __repr__(self):
        return (f"VisionJacobianLens(d_model={self.d_model}, n_samples={self.n_samples}, "
                f"target_block={self.target_block}, "
                f"source_blocks=[{self.source_blocks[0]}..{self.source_blocks[-1]}])")

    def save(self, path, *, dtype=torch.float16):
        torch.save({"J": {l: J.to(dtype) for l, J in self.jacobians.items()},
                    "n_samples": self.n_samples, "source_blocks": self.source_blocks,
                    "d_model": self.d_model, "target_block": self.target_block,
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path):
        c = torch.load(path, map_location="cpu", weights_only=False)
        return cls(jacobians=c["J"], n_samples=c["n_samples"], d_model=c["d_model"],
                   target_block=c["target_block"], meta=c.get("meta", {}))

    def transport(self, residual, block):
        """Map a block's pre-merge residual ``[..., 1024]`` into the final block's basis."""
        J = self.jacobians[block].to(device=residual.device, dtype=residual.dtype)
        return residual @ J.T


class VisionJLens:
    """A vision-tower J-Lens bound to a Qwen3.5 `Qwen3_5VisionModel` + tied unembed.

    Attributes:
        visual: the `Qwen3_5VisionModel` (24 blocks, merger inside).
        merger: `visual.merger` (LayerNorm(1024) -> 2x2 reshape -> fc1 -> GELU -> fc2 -> 2560).
        unembed_weight: ``[vocab, 2560]`` tied embed matrix (float32, on device).
        image_processor / tokenizer: Qwen image processor + tokenizer.
        merge_unit: 4. n_blocks: 24. d_vision: 1024. d_model: 2560.
        lens: fitted `VisionJacobianLens` or None.
    """

    def __init__(self, visual, merger, unembed_weight, image_processor, tokenizer,
                 *, lens=None, device="cuda:0"):
        self.visual = visual
        self.merger = merger
        self.unembed_weight = unembed_weight
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
        self.lens = lens
        self.n_blocks = len(visual.blocks)
        self.merge_unit = visual.spatial_merge_size ** 2
        self.d_vision = merger.norm.normalized_shape[0]
        self.d_model = unembed_weight.shape[1]

    # ---------- construction ----------
    @classmethod
    def from_pretrained(cls, model_id="Qwen/Qwen3.5-4B", *, device="cuda:0",
                        dtype=torch.bfloat16, lens=None):
        """Build from a single (public) Qwen3.5 VLM checkpoint — the reproducible path.

        Loads only the vision tower + merger (``model.visual.*``) and the tied
        unembedding (``model.language_model.embed_tokens.weight``) from ``model_id``;
        the LLM decoder is never materialised. ``model_id`` may be a HF hub id
        (e.g. ``"Qwen/Qwen3.5-4B"``) or a local directory. Only the shards holding
        those tensors are read, so peak host memory stays modest.

        `lens` may be a path to a saved `VisionJacobianLens`, a lens object, or None.
        """
        from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        from safetensors import safe_open
        import json

        def _resolve(fn):
            if os.path.isdir(model_id):
                return os.path.join(model_id, fn)
            from huggingface_hub import hf_hub_download
            return hf_hub_download(model_id, fn)

        cfg = AutoConfig.from_pretrained(model_id)
        vc = cfg.vision_config
        vc._attn_implementation = "sdpa"          # flash-attn not required; sdpa differentiable
        visual = Qwen3_5VisionModel(vc)

        idx = json.load(open(_resolve("model.safetensors.index.json")))
        wm = idx["weight_map"]
        emb_key = "model.language_model.embed_tokens.weight"
        need = {k: s for k, s in wm.items() if ".visual." in k or k == emb_key}
        by_shard = {}
        for k, s in need.items():
            by_shard.setdefault(s, []).append(k)
        vis_sd, emb = {}, None
        for shard, keys in by_shard.items():
            with safe_open(_resolve(shard), framework="pt", device="cpu") as f:
                for k in keys:
                    if k == emb_key:
                        emb = f.get_tensor(k)
                    else:
                        vis_sd[k.replace("model.visual.", "")] = f.get_tensor(k)
        if emb is None:
            raise RuntimeError(f"{model_id} has no {emb_key}; not a Qwen3.5 VLM checkpoint?")
        missing, unexpected = visual.load_state_dict(vis_sd, strict=False)
        # non-persistent buffers (e.g. rotary inv_freq) may be 'missing' — that's fine;
        # 'unexpected' keys mean a real mismatch.
        if unexpected:
            raise RuntimeError(f"unexpected vision keys: {unexpected[:6]}")
        visual = visual.to(device=device, dtype=dtype).eval()
        visual.requires_grad_(False)              # Jacobian is d(act)/d(act) at fixed weights
        unembed_weight = emb.to(device=device, dtype=torch.float32)

        ip = AutoImageProcessor.from_pretrained(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        if isinstance(lens, str):
            lens = VisionJacobianLens.load(lens)
        return cls(visual, visual.merger, unembed_weight, ip, tok, lens=lens, device=device)

    @classmethod
    def from_qwen35(cls, *, awq_snapshot, vis_weights, device="cuda:0",
                    dtype=torch.bfloat16, lens=None):
        """Build from the AWQ snapshot (unquantized vision tower + merger + tied embed).

        `awq_snapshot`: a `QuantTrio/Qwen3.5-4B-AWQ` snapshot dir (config, tokenizer, image
        processor, and the shard holding `model.language_model.embed_tokens.weight`).
        `vis_weights`: safetensors with the (unquantized) `model.visual.*` state -- the
        vision tower + merger (e.g. the reconstructed vis_only file).
        """
        from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        from safetensors.torch import load_file
        from safetensors import safe_open
        import json

        cfg = AutoConfig.from_pretrained(awq_snapshot)
        vc = cfg.vision_config
        vc._attn_implementation = "sdpa"          # flash-attn not installed; sdpa differentiable
        visual = Qwen3_5VisionModel(vc)
        sd = {k.replace("model.visual.", ""): v for k, v in load_file(vis_weights).items()}
        missing, unexpected = visual.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"vision weight load mismatch: missing={missing[:4]} "
                               f"unexpected={unexpected[:4]}")
        visual = visual.to(device=device, dtype=dtype).eval()
        visual.requires_grad_(False)              # Jacobian is d(act)/d(act) at fixed weights

        idx = json.load(open(os.path.join(awq_snapshot, "model.safetensors.index.json")))
        shard = idx["weight_map"]["model.language_model.embed_tokens.weight"]
        with safe_open(os.path.join(awq_snapshot, shard), framework="pt", device="cpu") as f:
            emb = f.get_tensor("model.language_model.embed_tokens.weight")
        unembed_weight = emb.to(device=device, dtype=torch.float32)

        ip = AutoImageProcessor.from_pretrained(awq_snapshot)
        tok = AutoTokenizer.from_pretrained(awq_snapshot)
        if isinstance(lens, str):
            lens = VisionJacobianLens.load(lens)
        return cls(visual, visual.merger, unembed_weight, ip, tok, lens=lens, device=device)

    # ---------- inputs ----------
    def preprocess(self, image, size=448):
        """PIL image (or path) -> (pixel_values[bf16, device], grid_thw[device], (rows,cols))."""
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if size is not None:
            image = image.convert("RGB").resize((size, size))
        proc = self.image_processor(images=[image], return_tensors="pt")
        pv = proc["pixel_values"].to(self.device, self.visual.dtype)
        gthw = proc["image_grid_thw"].to(self.device)
        t, hp, wp = [int(x) for x in gthw[0].tolist()]
        ms = self.visual.spatial_merge_size
        return pv, gthw, (hp // ms, wp // ms)

    # ---------- forward: capture every block residual ----------
    def _block_residuals(self, pv, gthw, *, grad=False):
        """Run the vision tower -> ({block: h_l [S,1024]}, y [P,2560]).

        With ``grad=True`` the input requires grad and the graph is retained, so returned
        tensors are differentiable (for Jacobian fitting)."""
        store = {}
        handles = [b.register_forward_hook(
            (lambda i: (lambda m, inp, out: store.__setitem__(i, out)))(i))
            for i, b in enumerate(self.visual.blocks)]
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                out = self.visual(pv, grid_thw=gthw)
            hs = {i: store[i] for i in range(self.n_blocks)}
            y = out.pooler_output
        finally:
            for h in handles:
                h.remove()
        return hs, y

    # ---------- readouts ----------
    def _unembed(self, y):
        """[..., 2560] -> [..., vocab] logits via the tied unembed (float32)."""
        return y.float() @ self.unembed_weight.T

    def naive_logits(self, h_block):
        """Naive vision-logit-lens: unembed(merger(h_l)) -> [P, vocab]."""
        return self._unembed(self.merger(h_block.to(self.visual.dtype)))

    def jacobian_logits(self, h_block, block):
        """Fitted vision-Jacobian-lens: unembed(merger(J_l @ h_l)) -> [P, vocab].

        At the target block the transport is the identity (``J = I``), so no ``J`` is
        stored for it; there the Jacobian readout is exactly the naive readout.
        """
        if self.lens is None:
            raise RuntimeError("no fitted lens: call .fit(images) or pass lens=")
        if block not in self.lens.jacobians:      # target block: J = I -> naive readout
            return self.naive_logits(h_block)
        transported = self.lens.transport(h_block.float(), block).to(self.visual.dtype)
        return self._unembed(self.merger(transported))

    def read_image(self, image, *, use_jacobian=False, blocks=None, size=448):
        """Spatial lens map -> dict(logits={block: [P, vocab]}, rows, cols).

        Reshape a block's logits to (rows, cols) for a heatmap over the merged patch grid.
        """
        pv, gthw, (rows, cols) = self.preprocess(image, size=size)
        hs, _ = self._block_residuals(pv, gthw, grad=False)
        blocks = list(blocks) if blocks is not None else list(range(self.n_blocks))
        out = {b: (self.jacobian_logits(hs[b], b) if use_jacobian else self.naive_logits(hs[b]))
               for b in blocks}
        return {"logits": out, "rows": rows, "cols": cols}

    def token_ids(self, words):
        """Single-token vocab ids for `words` (with/without leading space, cased)."""
        s = set()
        for w in words:
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                e = self.tokenizer.encode(v, add_special_tokens=False)
                if len(e) == 1:
                    s.add(e[0])
        return sorted(s)

    def object_heatmap(self, image, object_words, *, block, use_jacobian=False, size=448):
        """[rows, cols] map of the max lens score over `object_words`' token ids (+ the ids)."""
        res = self.read_image(image, use_jacobian=use_jacobian, blocks=[block], size=size)
        ids = self.token_ids(object_words)
        if not ids:
            raise ValueError(f"no single-token id for any of {object_words}")
        lg = res["logits"][block][:, ids].max(dim=1).values      # [P]
        return lg.view(res["rows"], res["cols"]), ids

    # ---------- fitting ----------
    def fit(self, images, *, R=64, size=448, source_blocks=None, verbose=True):
        """Fit `VisionJacobianLens` J_l = E[dh_23/dh_l] over `images` (paths or PIL).

        Each image contributes its per-block Jacobian at one replicated forward pass (the
        forward is repeated ``R`` times as separate images so attention stays intra-image and
        ``R`` output channels are done per backward pass); the running token+image mean is the
        lens. Stores and returns it.
        """
        target_block = self.n_blocks - 1
        source_blocks = (list(source_blocks) if source_blocks is not None
                         else list(range(target_block)))     # 0..22 (block 23 -> identity)
        acc = RunningJacobianAccumulator(source_blocks, self.d_vision, target_block)
        peak = 0.0
        for i, img in enumerate(images):
            pv, gthw, (rows, cols) = self.preprocess(img, size=size)
            pv_r = pv.repeat(R, 1)
            gthw_r = gthw.repeat(R, 1)
            torch.cuda.reset_peak_memory_stats(self.device)
            pv_r = pv_r.clone().requires_grad_(True)
            hs, _ = self._block_residuals(pv_r, gthw_r, grad=True)
            src = {b: hs[b] for b in source_blocks}           # packed [R*S, 1024] graph nodes
            per_img = fit_block_jacobian_packed(src, hs[target_block], R=R)
            norm = acc.add(per_img)
            del hs, src, per_img
            torch.cuda.empty_cache()
            gb = torch.cuda.max_memory_allocated(self.device) / 1e9
            peak = max(peak, gb)
            if verbose:
                name = os.path.basename(img) if isinstance(img, str) else f"img{i}"
                print(f"[vfit] {i + 1}/{len(images)} {name}  "
                      f"max||J||/sqrt(d)={norm:.3f}  peak {gb:.1f} GB")
        lens = acc.finalize(meta={"R": R, "size": size, "peak_gb": peak,
                                  "n_images": len(images)})
        # RunningJacobianAccumulator returns a BlockJacobianLens; wrap as VisionJacobianLens.
        self.lens = VisionJacobianLens(lens.jacobians, n_samples=lens.n_samples,
                                       d_model=self.d_vision, target_block=target_block,
                                       meta=lens.meta)
        return self.lens
