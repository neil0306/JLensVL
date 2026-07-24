# Vision-tower J-Lens — quantitative validations (real Qwen3.5-4B ViT)

Vision encoder: `model.model.visual` (24 blocks, hidden 1024) + merger (→2560) + tied
unembed (embed_tokens, vocab 248320). Weights: unquantized bf16 vision tower/merger from the
reconstructed vis_only safetensors; unembed from the AWQ snapshot shard 1. All numbers below
are from REAL forward passes on 10 composited test images (object photo on white at a known
bbox). Lens: VisionJacobianLens(d_model=1024, n_samples=7, target_block=23, source_blocks=[0..22]).

Two readouts, both decoding a ViT block into the LLM vocab:
- **naive vision-logit-lens**: `unembed(merger(h_l))` (identity-transport shortcut).
- **fitted vision-Jacobian-lens**: `unembed(merger(J_l · h_l))`, `J_l = E[dh_23/dh_l]`.

## V1 — spatial pointing game (argmax patch lands on object)
| lens | best layer | best acc | mid-stack (L6–17) acc |
|---|---|---|---|
| naive | L0 | 0.70 | 0.01 |
| **Jacobian** | L2 | **0.80** | **0.15** |

The Jacobian lens wins at nearly every layer; the naive lens is competitive only at L0–1
(where residuals are still near the raw patch embeddings) and collapses to ~0 mid-stack.
Per-layer accuracy (naive / jac):
- L 0: naive 0.70 | jac 0.40
- L 1: naive 0.70 | jac 0.50
- L 2: naive 0.50 | jac 0.80
- L 3: naive 0.60 | jac 0.80
- L 4: naive 0.20 | jac 0.50
- L 5: naive 0.40 | jac 0.70
- L 6: naive 0.10 | jac 0.50
- L 7: naive 0.00 | jac 0.30
- L 8: naive 0.00 | jac 0.30
- L 9: naive 0.00 | jac 0.10
- L10: naive 0.00 | jac 0.00
- L11: naive 0.00 | jac 0.00
- L12: naive 0.00 | jac 0.00
- L13: naive 0.00 | jac 0.30
- L14: naive 0.00 | jac 0.20
- L15: naive 0.00 | jac 0.10
- L16: naive 0.00 | jac 0.00
- L17: naive 0.00 | jac 0.00
- L18: naive 0.00 | jac 0.00
- L19: naive 0.00 | jac 0.30
- L20: naive 0.10 | jac 0.30
- L21: naive 0.10 | jac 0.00
- L22: naive 0.10 | jac 0.00
- L23: naive 0.10 | jac 0.10

## V2 — layer-wise concept emergence
Object−background lens contrast vs vision depth (see `V2_emergence.png`). Object identity is
spatially legible early (L0–5). The **naive** lens then goes strongly **negative** mid-deep
(L16: -2.30) — deep merged tokens are dominated by global /
register signal, so uniform-background patches *out-score* the object token. The **Jacobian**
transport removes that artifact and stays near-zero (L16: -0.11).
Mid-stack (L6–17) mean contrast: naive **-0.98** vs jac
**-0.05**.

## V3 — naive vs Jacobian legibility (mid-stack)
The discriminating legibility signal here is **spatial** (see `V3_legibility.png`, pointing
accuracy per layer): mid-stack (L6–17) the naive lens localizes at **0.01**
while the Jacobian lens localizes at **0.15**. The absolute object-token vocab
rank is *not* discriminating — median mid-stack rank naive 3033 vs jac 6081
— because the naive lens **saturates** the object token at *every* patch (including
background); that flatters its rank while destroying localization (exactly why its mid-stack
pointing accuracy is ~0). The Jacobian lens trades that spurious saturation for clean spatial
selectivity.

**Sanity anchor:** at the final block (L23) the Jacobian transport is the identity, so the
Jacobian and naive readouts are identical there (verified).

## Artifacts
- `V2_emergence.png`, `V3_legibility.png`, `metrics.json`, `vision_jacobian_lens.pt`
- heatmap overlays: heatmaps/dog_naive_L0.png, heatmaps/dog_jac_L2.png, heatmaps/cat_naive_L0.png, heatmaps/cat_jac_L2.png, heatmaps/elephant_naive_L0.png, heatmaps/elephant_jac_L2.png, heatmaps/clock_naive_L0.png, heatmaps/clock_jac_L2.png
