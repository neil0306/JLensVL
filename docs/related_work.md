# Related work — the J-Lens / VLM-lens landscape (and where JLensVL sits)

Compiled 2026-07-11. Scope: everything around the Jacobian lens (J-Lens) and
lens-style reading of vision-language models, and an honest positioning of
JLensVL against it. See also `docs/native_visual_lens_design.md` for the
cross-modal / encoder-side design this motivates.

## 0. TL;DR positioning
JLensVL is one of only ~3 projects that actually **read a J-Lens on a VLM**, and
it goes furthest on the **causal** axis (concept-race + lens-coordinate swap
interventions on Qwen3.5-4B, with a recorded decision-flipping result). What the
deeper work has that we don't (yet): peer review + multi-model scale
(**LatentLens**), a ground-truth-benchmarked *reading* method
(**LatentLens** retrieval), a formal correlational-vs-causal lens contrast
(**Nanda review** vs tuned lens), pre-registered statistical rigor
(**solarkyle/jspace**), and an **encoder-side / cross-modal** lens
(jerrickhoang stub + LatentLens). Our two active tracks (retrieval-lens for
Qwen3.5; cross-modal Jacobian) are aimed precisely at closing those gaps.

## 1. Foundation
- **anthropics/jacobian-lens** — https://github.com/anthropics/jacobian-lens
  (~1.1k★, Apache-2.0, unmaintained). Paper: *"Verbalizable Representations Form
  a Global Workspace in Language Models,"* transformer-circuits.pub, 2026-07-06.
  Defines `lens_ℓ(h)=unembed(J_ℓ·h)`, `J_ℓ=E[∂h_final/∂h_ℓ]`, fitted over ~1000
  seqs×128 tok via a cotangent-summation VJP estimator. J-space (~6–10% of
  activation variance) acts as a "global workspace" for verbalizable concepts.
  **Text-only, decoder-only.** Pre-fitted lenses on Neuronpedia (Gemma-2/3).
  → This is the engine JLensVL builds on; the VLM extension is our gap-fill.

## 2. Direct VLM competitors (projects that read a lens on a VLM)
- **jerrickhoang/jlens-qwen-jspace** — https://github.com/jerrickhoang/jlens-qwen-jspace
  (0★, Qwen3.5-9B, H100). **Closest sibling.** Same research question as us
  ("do image tokens enter a language-defined J-space at vision vs post-translation
  positions?"). Measures vision-token energy in J-space via top-k SVD /
  effective-rank pre-pass; has a **mandatory text-only sanity gate** (baseline
  0.405 top-1) and swap/suppress interventions; ships an **encoder-side stub**
  (registered, unimplemented). Deeper *methodology* than us on the sanity/rank
  side; we're ahead on the causal-intervention framing (concept-race + validated
  swap).
- **jude-sph/J-lens-Vision** — https://github.com/jude-sph/J-lens-Vision
  (0★, LLaVA-1.5-7B). Companion: Hawrani, *"Reading into VLM hallucinations using
  the Jacobian lens,"* LessWrong 2026-07-10. Finds the workspace holds the correct
  answer but yes/no framing triggers a yes-bias override (39/39 false positives on
  a lamp probe; hallucinated-object rank ~111 vs real ~16). Has causal
  workspace-editing (overlaps our interventions) but **explicitly informal**, 1
  model / 39 images.
- **idhantgulati/j-lens** — https://github.com/idhantgulati/j-lens (1★). Same J_ℓ
  framework on **Qwen3.5-4B (our exact model)**, `interventions.py` doing the same
  steering/ablation/coordinate-swap family. **Text-only** — useful as a second
  reference impl of causal swaps, not a VLM competitor.
- **WeZZard/jlens-qwen36** — https://github.com/WeZZard/jlens-qwen36 (302★).
  Apple-MLX port of Qwen3.6-27B; custom Metal backward kernel (22× speedup),
  63-layer fit in 2.75h on M4 Pro. **Despite branding, NOT a VLM** ("visual
  debugger" = the UI). Engineering-deep, not a lens-reading competitor.

## 3. The "theoretically deeper" work (what motivated going deeper)
Two senses of "deeper" — keep them separate:

### 3a. Deeper VLM-specific lens work
- **LatentLens (McGill-NLP), ICML 2026** — paper
  https://arxiv.org/abs/2602.00462 · repo https://github.com/McGill-NLP/latentlens
  (46★, MIT). **The only peer-reviewed entry.** Reframes lens-decoding as
  **retrieval, not projection**: reads a hidden state by nearest-neighbor lookup
  in a contextual-embedding index (~117k sentences / ~23k concepts) instead of a
  `W_U` multiply. Formally contrasts EmbeddingLens / LogitLens (and by extension
  Jacobian/tuned lenses). Evaluated on **15 VLMs at all layers**; headline claim:
  **LogitLens "substantially underestimates" the interpretability of visual
  tokens** — a direct, benchmarked critique of exactly the projection approach
  J-Lens/JLensVL use. Ships pre-built indices incl. Qwen2.5-VL-7B.
  - **Our stance (important):** LatentLens's shipped indices/vision stack are
    **Qwen2.5-VL**, whose vision tower has a *different training set and prior
    distribution* from our **Qwen3.5**. Using their Qwen2.5-VL readout directly
    as ground truth for our model is **not apples-to-apples**. Even peer-reviewed,
    we treat it as a **reference, not GT**. Plan: **reimplement the LatentLens
    *method* to fit a retrieval lens on Qwen3.5 itself**, then use *that* to
    estimate how much our projection J-Lens misses — with the retrieval readout as
    a strong reference baseline, not an oracle. (See Track A in the task plan.)

### 3b. Deeper pure theory of J-Lens (no VLM)
- **Neel Nanda, *"A Review of Anthropic's Global Workspace Paper,"*** LessWrong
  2026-07-06 — https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/ . Most
  theoretically substantive public J-Lens treatment. Separates **J-Lens**
  (readout) from **J-space** (sparse combos of `J·W_U`). Sharpest tuned-lens
  contrast: tuned lens is *correlational* regression that "skips ahead" to
  downstream-computed concepts; J-Lens's infinitesimal-perturbation Jacobian is
  *causal-in-linear-approximation*, revealing raw contents. Proposes
  computation-as-causal-graph + a "coordination-problem" hypothesis for shared
  subspaces. **Zero VLM content** — but the correlational-vs-causal framing is a
  cheap, high-value formalization we can adopt (Track A add-on: tuned-lens
  contrast on our swap cases).
- **solarkyle/jspace** — https://github.com/solarkyle/jspace (37★, MIT). Deepest
  *empirical/statistical* rigor (text-only): pre-registered hallucination campaign
  25,340 prompts / 13 benchmarks, confound-checked stats, workspace-entropy AUROC
  0.789 vs logprob 0.731 (leave-one-dataset-out), honest failure reporting. The
  rigor bar for our eval track.

## 4. Adjacent / prior art
- **Tuned Lens** (Belrose 2023) — https://arxiv.org/abs/2303.08112 · repo
  https://github.com/AlignmentResearch/tuned-lens . Learned per-layer affine probe
  minimizing KL; J-Lens positioned as its causal refinement.
- **Patchscopes** (Ghandeharioun 2024) — https://arxiv.org/abs/2401.06102 . Patch
  a hidden state into another forward pass and read the output; theoretical
  generalization of logit-lens inspection.
- **Jacobian Sparse Autoencoders (JSAEs)** — https://arxiv.org/abs/2502.18147
  (ICML 2025). Sparsifies *computation* (Jacobian between SAE latents), not
  activations. Text-only; prior art on "why Jacobians are a principled
  interpretability object."
- **ViT-Prisma / Prisma** — https://github.com/Prisma-Multimodal/ViT-Prisma ·
  https://arxiv.org/abs/2504.19475 . TransformerLens-for-ViTs toolkit (75+ models,
  vision SAEs, logit-lens for ViTs). Infrastructure we could interoperate with for
  the encoder-side lens, not a theory competitor.
- **V-SEAM** (2509.14837), **"Towards Interpreting Visual Information Processing in
  VLMs"** (2410.07149), **VLM-Lens** (2510.02292) — general VLM
  causal-interp / activation-patching; relevant related work for our causal
  lens-swap track, not J-Lens-specific.

## 5. Where JLensVL uniquely leads / trails
| Axis | JLensVL | Best-in-class elsewhere |
|------|---------|-------------------------|
| Reads J-Lens on a VLM | ✅ (image / post-image / answer positions) | jerrickhoang, jude-sph |
| Causal lens-coordinate swap on VLM | ✅ validated decision-flip (Qwen3.5-4B) | jude-sph (informal) |
| Concept-race / contradiction analysis | ✅ | — (differentiator) |
| Peer review + multi-model scale | ❌ 1 model | LatentLens (15 VLMs, ICML) |
| Ground-truth-benchmarked *reading* | ❌ | LatentLens (retrieval) |
| Formal correlational-vs-causal / tuned-lens contrast | ❌ (asserted, not shown) | Nanda review |
| Pre-registered statistical rigor | ❌ (anecdotal) | solarkyle/jspace |
| Encoder-side / cross-modal lens | ⏳ designed, unbuilt | jerrickhoang (stub), LatentLens |

## 6. Concrete "go deeper" moves (feed the task plan)
1. **Fit a LatentLens-style retrieval lens on Qwen3.5 itself** (not reuse their
   Qwen2.5-VL index), then measure how much our projection J-Lens misses at image
   / post-image / answer positions — retrieval as a *reference baseline*.
2. **Build the cross-modal Jacobian / encoder-side lens** per
   `docs/native_visual_lens_design.md` (native `Jv` + cross-modal `Jx`, in-encoder
   vs fusion-born comparison figure).
3. **Formalize the tuned-lens contrast** (Nanda): fit a tuned lens on the same
   layers, show on our swap cases that J-Lens reveals raw pre-processing content
   where tuned lens "skips ahead."
4. **Statistical campaign** (solarkyle): scale from one decision-flip to a
   pre-registered set of image/prompt pairs with AUROC / leave-one-out reporting.
5. **Adopt jerrickhoang's sanity gate + effective-rank pre-pass** to guard against
   reading noise as "visual concepts in J-space."
