# Design: native visual lens + cross-modal Jacobian for JLensVL

Status: **design / not yet implemented** (needs model weights resident + a new
adapter + a fitting pass). Grounded in the real Qwen3.5-4B module layout.

## Motivation (the gap this closes)
JLensVL today reads only the **LLM-side** residual stream (post-tokenization),
treating the vision tower as a black box. We cannot currently answer: *is a
concept already linearly readable inside the vision encoder, or does it only
emerge after the LLM fuses vision + text?* Two new observers close this:

1. **Native visual lens** — a Jacobian lens fitted *directly on the ViT patch
   stream*, reading a vision-block residual into a decodable basis.
2. **Cross-modal Jacobian** — a single Jacobian from a **ViT patch output**
   straight through merger + all 32 LLM layers to the LLM final residual, so a
   patch can be read as vocab concepts end-to-end.

## Confirmed Qwen3.5-4B layout (`Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`)
- Vision tower: `model.visual`
  - `model.visual.blocks[0..23]` — 24 ViT blocks, hidden **1024**, patch 16,
    `spatial_merge_size=2`, `temporal_patch_size=2`, 16 heads.
  - `model.visual.merger` — MLP (`linear_fc1`→`linear_fc2`) projecting merged
    patches to `out_hidden_size=2560` (the LLM residual width).
  - `deepstack_visual_indexes` in config: selected visual-block outputs are also
    injected at several LLM layers (deepstack). **This matters** — see caveats.
- LLM: `model.language_model.layers[0..31]`, hidden **2560**, final norm +
  `lm_head` (vocab 248320). `image_token_id=248056`.

The existing text/LLM lens already targets `model.language_model` via the engine
layout `Layout("model.language_model")`; nothing here changes that.

## Method
Let `p_l` = residual of a ViT patch at vision block `l` (dim 1024). Define, by
analogy to the LLM lens `J_L = E[∂h_final/∂h_L]`:

- **Native visual lens** `Jv_l = E[ ∂ m / ∂ p_l ]` where `m` is the **merger
  output** for that patch (dim 2560, the LLM-input basis). Read a patch by
  `unembed( Jv_l · p_l )` — reuse the LLM's own `final_norm + lm_head` as the
  decoder (the merger output already lives in the 2560 LLM basis). This shows
  what vocab concept a patch is poised to contribute *before* the LLM runs.
  Averaged over patches and over a fitting image set (like the LLM lens averages
  over token positions/prompts).

- **Cross-modal Jacobian** `Jx = E[ ∂ h_final^LLM / ∂ p_L ]` — from a chosen ViT
  block `L` all the way to the LLM final residual, at a chosen readout token
  position (the answer position, or the image-token span). Read by the normal
  LLM `unembed`. Comparing `unembed(Jv_L·p)` (in-encoder) vs `unembed(Jx·p)`
  (post-fusion) tells you whether a concept is native to vision or fusion-born.

Both are forward-only at inference once fitted, exactly like the LLM lens.

## Reuse vs. new code
Reuse: `ActivationRecorder` works on any `nn.ModuleList` → point it at
`model.visual.blocks`; its `start_graph_at` already supports rooting the autograd
graph at a captured activation (needed so the Jacobian fit only spans block `l`
onward). Reuse `JacobianLens` storage/`transport`/`save`/`merge` unchanged — a
`Jv`/`Jx` is just a `{layer: [d_out, d_in]}` matrix set (note: **non-square**
for the native lens, 2560×1024, so `JacobianLens` must relax its implicit square
assumption — small change, or store as a separate `RectJacobianLens`).

New code needed:
1. `VisionLensModel` adapter (mirrors `HFLensModel`): exposes
   `.layers = model.visual.blocks`, `.n_layers=24`, `.d_model=1024`, an
   `encode_image(pixel_values, grid_thw)` that runs `model.visual` up to the
   readout, and an `unembed` that routes merger-output → LLM `final_norm+lm_head`.
   The vision forward needs `pixel_values` + `image_grid_thw` from the processor,
   not `input_ids` — so `encode`/`forward` signatures differ from the text path.
2. A fitting entry `fit_visual(vlm, images, at_blocks=...)` that captures patch
   residuals with grads and averages `∂m/∂p_l` (native) or `∂h_final/∂p_L`
   (cross-modal) over patches × images. Can largely follow `jlens.fitting.fit`'s
   per-dimension Jacobian accumulation, swapping the input leaf.
3. `JLensVL.trace_visual(image, question, blocks=...)` readout method + a
   `viz.visual_slice_grid_html` (patch-grid heatmap over the image, per block).

## Caveats / decisions to make at implementation time
- **Variable patch count**: patch/grid size depends on image resolution. Fit the
  Jacobian *per-patch then average* (patch is the analogue of token position);
  keep images at a fixed processed resolution during fitting for a clean average.
- **spatial_merge_size=2**: the merger consumes 2×2 patch groups. Decide the unit
  of readout — pre-merge patch (1024) vs post-merge token (2560). Native lens as
  defined targets the merger *output* token, cleanest as post-merge.
- **deepstack injection**: some visual features re-enter the LLM at deeper layers,
  so a pure `Jx` from block L to LLM-final understates influence via the deepstack
  path. Document this; optionally fit separate `Jx` per deepstack entry point.
- **dtype/grad**: vision tower runs bf16; cast the captured patch leaf to fp32 and
  `requires_grad_(True)` (via `ActivationRecorder(start_graph_at=L)`), keep model
  params frozen (already done by `HFLensModel`; replicate in the vision adapter).
- **Memory**: 24 ViT blocks × Jacobian fit is cheaper than the LLM (1024 dim), but
  cross-modal `Jx` spans 24+32 blocks — fit on the RTX PRO 6000 (GPU 0) when free,
  or the 3090 with gradient-checkpointed blocks and a small image batch.

## Phasing
- **P1**: `VisionLensModel` adapter + capture-only `trace_visual` using the plain
  logit-lens (no Jacobian) as a first legibility check — cheap, no fitting.
- **P2**: fit the **native visual lens** `Jv` (single block first, then a sweep),
  add the patch-grid viz. Verify a known concept (e.g. a pug image) is readable
  in-encoder at some block.
- **P3**: fit the **cross-modal Jacobian** `Jx`; produce the in-encoder-vs-fusion
  comparison figure that is the headline result of this track.
- **P4**: extend the causal `LensIntervention` to patch at `model.visual.blocks`
  (swap a concept's *visual* lens coordinate) — the vision-side analogue of the
  LLM-side swap already implemented.
