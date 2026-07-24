# Theory of the Jacobian Lens for vision-language models (JLensVL)

Compiled 2026-07-11. Scope: the theoretical foundation under JLensVL — what the
J-Lens is as a mathematical object, what it does and does not license us to
claim, and how those limits shape the prompt-helper and the cross-modal design.
This document complements `docs/related_work.md` (the landscape and positioning)
and `docs/native_visual_lens_design.md` (the encoder-side construction); it does
not repeat the survey of competing projects. Its purpose is to make the tool
*trustworthy* — every readout the tool produces is a claim, and this document
states precisely how strong that claim is.

## Abstract

The Jacobian lens reads an internal activation `h_ℓ` by transporting it into the
final-layer basis with the model's *average input→output Jacobian*
`J_ℓ = E[∂h_final/∂h_ℓ]` and decoding it with the unembedding:
`lens_ℓ(h) = unembed(J_ℓ · h)`. `J_ℓ` is fitted once, offline, over a corpus;
at inference the lens is forward-only — one forward pass plus a matmul, no
per-query backprop. We give the formalism (§1), the correlational-vs-causal
distinction that separates the J-Lens from the tuned lens (§2), and the single
most important caveat we carry from sibling work: a lens *readout* is a
sensitivity-flavoured observation, not an attribution, so an observational lens
score is a **hypothesis**, not proof, and must be confirmed by a causal
intervention or by real task accuracy (§3). Sections 4–5 apply this to the
prompt helper and to the native-visual / cross-modal lenses; §6 states the scope
honestly. The recurring theme: the J-Lens is a principled, cheap
hypothesis-*generator*; trust in any specific claim comes from the causal or
empirical confirmation that follows it, never from the lens score alone.

## 1. The J-Lens formalism

Let a decoder transform a residual stream `h_ℓ ∈ ℝ^d` at layer `ℓ` into a final
residual `h_final ∈ ℝ^d`, then a fixed readout `unembed(·) = W_U · norm(·)`
maps `ℝ^d → ℝ^|V|` (final norm folded into the unembedding). The **plain logit
lens** applies `unembed` directly to `h_ℓ`, i.e. it *skips the transport* from
layer `ℓ` to the final layer and pretends the mid-stack activation already lives
in the output basis. On models like Qwen3.5 this is empirically noise mid-stack
(README §Why: at L20 the logit lens returns `baku / 魄 / ernen` while the true
concept is `currency`): the residual basis rotates and rescales across depth, so
reading `h_ℓ` in the final basis without transporting it reads the wrong
coordinates.

The J-Lens inserts the missing transport. Define the **average Jacobian**

```
J_ℓ = E_x[ ∂h_final / ∂h_ℓ ] ∈ ℝ^{d×d},
```

the expectation of the input→output Jacobian of the layer-`ℓ`→final map, taken
over activations `x` drawn from a fitting corpus. The lens readout is

```
lens_ℓ(h) = unembed( J_ℓ · h ).
```

`J_ℓ · h` is the first-order (linear) image of `h` under the network's own
layer-`ℓ`→final computation: it answers "if the activation at `ℓ` were `h`, what
final residual does the network's *local linearisation* send it to?" Because the
Jacobian carries the layer's real linear response, mid-stack activations become
legible — the transport undoes the basis rotation that defeats the logit lens.

**Fitting.** `J_ℓ` is estimated once by a cotangent-summation
vector–Jacobian-product (VJP) estimator over a corpus (Anthropic's reference
engine: ~1000 sequences × 128 tokens). One accumulates, per output coordinate,
the VJP `vᵀ (∂h_final/∂h_ℓ)` for basis cotangents `v` and averages over corpus
positions, yielding the `d×d` matrix (per layer). This is the only place a
backward pass is used. J-space — the sparse combinations of `J_ℓ · W_U` that
carry verbalizable concepts — occupies roughly 6–10% of activation variance and
behaves as a "global workspace" (Anthropic 2026).

**Inference is forward-only.** With `{J_ℓ}` stored, every query is: run one
forward pass to obtain `h_ℓ`, compute `unembed(J_ℓ · h_ℓ)`. No gradients, no
backprop per query — the property that makes the MLX Apple-Silicon backend
possible without a custom backward kernel (README §MLX).

## 2. Correlational vs. causal: the tuned-lens contrast

The tuned lens (Belrose 2023) also transports mid-stack activations into the
output basis, but it learns a **per-layer affine map** `A_ℓ h + b_ℓ` by
regression, minimizing the KL between its readout and the model's true final
distribution. This makes it *correlational*: the fitted `A_ℓ` is free to exploit
any statistical predictor of the final answer available at layer `ℓ`, including
information about what **downstream** layers will compute. As Neel Nanda's review
of the global-workspace paper puts it, the tuned lens can "skip ahead" —
surfacing a concept that is not yet present in `h_ℓ` but that later layers will
produce, because that concept is *correlated* with `h_ℓ` across the training
distribution.

The J-Lens uses no learned regression. `J_ℓ = E[∂h_final/∂h_ℓ]` is an actual
infinitesimal-perturbation Jacobian of the network's own function: it reports how
the final residual *actually responds* to a small change in `h_ℓ`, averaged over
inputs. It is therefore **causal in the linear approximation** — it reveals the
raw contents of the activation *at that layer*, i.e. the concepts that layer `ℓ`
itself is already carrying and that the downstream computation will linearly
propagate, rather than concepts that the downstream computation will freshly
manufacture.

**Practical consequence for interpretation.** When the J-Lens shows a concept at
layer `ℓ`, the honest reading is "this concept is *present in the activation at
`ℓ`* and linearly reaches the output," not merely "the answer is predictable from
`ℓ`." A tuned-lens sighting of the same concept does not distinguish the two.
This is why, for questions like "*when* does vision override the textual prior?"
(README §3, the dog/cat crossover), the J-Lens is the appropriate instrument: it
locates where the concept genuinely enters the stream, whereas a correlational
probe can report the concept early merely because it is predictable. The
distinction is what earns the J-Lens the word "causal" — with the crucial
qualifier of the next section.

## 3. Sensitivity vs. attribution — the caveat we adopt

"Causal in the linear approximation" is a claim about an *infinitesimal*
perturbation. It does not license claims about *finite* interventions, and
conflating the two is the central failure mode this document guards against.

This is a known caveat of Jacobian-norm interpretations in general: the raw
**Jacobian norm** — the *sensitivity* of the output to a coordinate — does
**not** reliably predict the damage caused by a **finite** perturbation of that
coordinate. The quantity that does predict
finite-perturbation effect is an **attribution**: the gradient *dotted with the
actual perturbation*,

```
attribution ≈ ⟨ ∂output/∂x , Δx ⟩       (grad × the real Δx / quantization error),
```

not `‖∂output/∂x‖` alone. A direction can be highly sensitive yet never be
perturbed in practice (small `Δx`), or be moderately sensitive but take a large
real `Δx`; sensitivity ranks these wrong. Sensitivity tells you *what the model
could react to*; attribution tells you *what actually moved the answer*.

The J-Lens readout `unembed(J_ℓ · h)` is built from `J_ℓ`, a sensitivity object
(an averaged Jacobian). A high J-Lens score for a concept is therefore a
**sensitivity-flavoured observation**: "this concept is linearly present and the
output is responsive to it here." It is *not* an attribution and *not* a
guarantee that intervening on that concept — or that a real change of phrasing —
will move the model's actual behavior by the amount the score suggests.

**Why this matters specifically for a prompt helper.** When JLensVL claims
"phrasing A makes the model perform better than phrasing B," the underlying
signal is a lens readout at the answer position — an *observational*,
sensitivity-flavoured quantity. By the argument above, that readout is a
**hypothesis**, not proof. Trusting the lens margin alone would be exactly the
sensitivity-vs-attribution error this caveat warns against. The correct
epistemics, and the theoretical justification for the prompt helper's staged
design, are:

- **P2 — causal validation.** Confirm the hypothesis with a real causal
  intervention: the existing lens-coordinate swap in `interventions.py` (swap the
  intended concept's J-space coordinate and check the decision actually flips, as
  in the recorded decision-flipping result). A swap is a *finite* intervention,
  so it measures attribution, not sensitivity.
- **P3 — real-accuracy bridge.** Confirm against the ground truth that ultimately
  matters: real task accuracy under phrasing A vs. B. If the lens margin and the
  accuracy delta agree, the observational signal is validated; if they diverge,
  the accuracy wins and the lens claim is retracted.

The lens is what makes P2/P3 cheap to *target* (it tells you which concept and
which layer to intervene on, forward-only, without a search); it is not a
substitute for them.

## 4. What the prompt helper measures — and its honesty condition

Read at the **answer position**, `lens_ℓ(h)` is naturally interpreted as *"what
the model is poised to say"* at depth `ℓ` — the concept currently occupying the
workspace before the token is emitted. The prompt helper turns this into a
comparison signal between phrasings by reading the **layer-wise concept
trajectory** of each candidate sense:

- **crystallization depth** — how early the intended sense becomes the top-1
  concept (earlier = the phrasing commits the model sooner);
- **crystallization strength** — the intended sense's score at the answer
  position;
- **competitor margin** — top-1(intended) − top-1(best competitor); a larger
  margin means the phrasing more decisively excludes the wrong sense (README §4:
  "In software engineering, Java is a" → programming margin +11.94 vs. the island
  phrasing's −7.50).

This is a **principled, forward-only** signal for word choice: earlier and
stronger crystallization with a smaller competitor margin is exactly what a
clearer prompt should induce in the workspace, and it is measurable in one
forward pass across all variants — no generation, no accuracy loop, no per-query
backprop.

**The honesty condition (ties back to §3).** The margin is *observational*. A
larger lens margin is evidence that phrasing A steers the workspace more
strongly, but it is a hypothesis about behavior, not a measurement of behavior.
The prompt helper is therefore correct to present the margin as a *ranking to be
confirmed*, and the tool's verdicts ("CLEAR", "OFF-TARGET") are calibrated
statements about the lens signal, not certified statements about task accuracy —
which is precisely why P2 (causal swap) and P3 (real accuracy) exist. Stated
plainly: the prompt helper generates the best hypotheses cheaply; it does not
retire them.

## 5. Native-visual lens and the cross-modal Jacobian

The same formalism extends to the vision side (full construction and the real
Qwen3.5-4B layout in `docs/native_visual_lens_design.md`; here we state the
theory and the evidence it works).

Let `p_ℓ` be the residual of a ViT patch at vision block `ℓ` (dim 1024 on
Qwen3.5-4B, 24 blocks). Two observers extend the LLM lens:

- **Native visual lens** `Jv_ℓ = E[∂m/∂p_ℓ]`, where `m` is the **merger output**
  for that patch (dim 2560, the LLM-input basis). Read a patch by
  `unembed(Jv_ℓ · p_ℓ)`, reusing the LLM's own `final_norm + lm_head` as decoder.
  This shows what vocab concept a patch is poised to contribute *before the LLM
  runs*. Note `Jv_ℓ` is **non-square** (2560×1024), so the square-Jacobian
  assumption of the LLM `JacobianLens` must be relaxed (or a `RectJacobianLens`
  stored).
- **Cross-modal Jacobian** `Jx = E[∂h_final^LLM/∂p_L]`, from a chosen ViT block
  `L` through the merger and all 32 LLM layers to the LLM final residual, read by
  the normal LLM `unembed`. Comparing `unembed(Jv_L · p)` (in-encoder) against
  `unembed(Jx · p)` (post-fusion) separates concepts **native to vision** from
  concepts that are **fusion-born** — i.e. only legible after the LLM combines
  vision and text.

Both are forward-only at inference once fitted, exactly like the LLM lens.

**Deepstack caveat.** Qwen3.5's `deepstack_visual_indexes` re-inject selected
visual-block outputs at several deeper LLM layers. A pure `Jx` from block `L` to
the LLM final therefore *understates* the total visual influence, because part of
that influence travels the deepstack path rather than the block-`L`→final path
being differentiated. The honest options are to document this understatement or
to fit a separate `Jx` per deepstack entry point.

**Evidence the approach works (fork validation on Qwen3.5-4B).** These numbers
are what let us claim the visual construction is sound rather than merely
plausible:

- **Pointing-game accuracy** (does the lens localize the named object to the
  right patches): fitted **Jacobian lens 0.80** vs. **naive logit lens 0.70** at
  the best layer — the Jacobian transport helps on the vision side too.
- **Mid-stack object-vs-background contrast** (mean over mid blocks): naive
  logit lens **−0.98** (essentially inverted — it reads background as strongly as
  object) vs. Jacobian **−0.05** (near-neutral, i.e. the transport removes the
  spurious mid-stack signal that fools the naive readout). This is the vision
  analogue of the text "logit lens is noise mid-stack" result.
- **Sanity anchor at the final vision block L23**: there the transport is the
  identity, so the Jacobian readout **equals** the naive readout exactly
  (**cosine 1.0**). This is the control that confirms the machinery is correct —
  where transport should be a no-op, it is.

The §3 caveat still applies here: a native-visual lens sighting of a concept in a
patch is an observation to be causally checked (the P4 vision-side lens-coordinate
swap in `native_visual_lens_design.md`), not a proof that the patch caused the
answer.

## 6. Limitations and scope

- **It is a linearization.** `J_ℓ` is an *averaged* Jacobian; `J_ℓ · h` is a
  first-order image. Nonlinearities and input-dependent Jacobian variation are
  discarded. Expect both **false positives** (a concept that the linearization
  surfaces but the full nonlinear network does not commit to) and **false
  negatives** (a concept present nonlinearly that the averaged linear map misses).
- **Observation, not attribution (§3).** Every readout is sensitivity-flavoured.
  It ranks and localizes hypotheses; it does not by itself measure the effect of a
  finite intervention or of a real prompt change.
- **Correlation with output, not identity with it.** The lens reads
  *poised-to-say* content, which correlates with — but does not equal — the
  emitted token (README §Caveats). Use aggregates and baselines, not single
  readouts, as evidence.
- **White-box only.** Needs weights plus a fitted lens; it cannot run against a
  closed API.
- **Corpus dependence.** `J_ℓ` and J-space are averaged over a fitting corpus; a
  lens fitted on one distribution may transport a different distribution less
  faithfully.

**Intended use.** The J-Lens is best used as a **hypothesis-generation tool**:
cheap, forward-only, and principled enough to point at the right concept, layer,
and patch — after which the hypothesis is confirmed by a causal intervention
(lens-coordinate swap) and/or by real task accuracy. Trust in any specific claim
comes from that confirmation, not from the lens score. No claim in JLensVL should
be stated more strongly than the stage of validation it has actually reached.

## References

- **Anthropic (2026), *Verbalizable Representations Form a Global Workspace in
  Language Models*** — transformer-circuits.pub/2026/workspace/. Defines
  `lens_ℓ(h)=unembed(J_ℓ·h)`, `J_ℓ=E[∂h_final/∂h_ℓ]`, the VJP fitting estimator,
  and J-space as a global workspace. Reference engine:
  https://github.com/anthropics/jacobian-lens (Apache-2.0).
- **Neel Nanda (2026), *A Review of Anthropic's Global Workspace Paper*** —
  LessWrong, https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/ . The
  correlational (tuned-lens, "skips ahead") vs. causal-in-linear-approximation
  (J-Lens) distinction adopted in §2, and the J-Lens / J-space separation.
- **Belrose et al. (2023), *Eliciting Latent Predictions with the Tuned Lens*** —
  https://arxiv.org/abs/2303.08112 · https://github.com/AlignmentResearch/tuned-lens .
  The learned per-layer affine probe the J-Lens is contrasted against in §2.
- **McGill-NLP (2026), LatentLens**, ICML 2026 —
  https://arxiv.org/abs/2602.00462 · https://github.com/McGill-NLP/latentlens .
  Reframes lens decoding as retrieval; benchmarks how much projection lenses miss
  on visual tokens — the reference (not ground truth) for the visual readout.
- Companion JLensVL docs: `docs/related_work.md` (landscape and positioning),
  `docs/native_visual_lens_design.md` (the native-visual + cross-modal
  construction and the real Qwen3.5-4B module layout).
