# JLensVL

**A Jacobian-Lens (J-Lens) observer for vision-language models — read what a model is _poised to say_, before it says it.**

The [Jacobian lens](https://transformer-circuits.pub/2026/workspace/) (Anthropic, 2026) reads an internal activation by transporting it into the final-layer basis with the model's average input→output Jacobian and decoding it with the unembedding:

```
lens_ℓ(h) = unembed( J_ℓ · h )      J_ℓ = E[ ∂h_final / ∂h_ℓ ]
```

It shows the concepts a model holds in its "global workspace" at each layer. **JLensVL brings the J-Lens to _vision-language_ models** — read the lens at image-token positions, post-image text positions, and the answer position — plus concept-competition ("race") analysis and a forward-only **prompt helper**.

> The lens is **forward-only at inference**: `J_ℓ` is fitted once, then every readout is one forward pass + a matmul. No backprop, no per-query gradients.

Built on Anthropic's reference [`jacobian-lens`](https://github.com/anthropics/jacobian-lens) engine (which fits `J_ℓ`); JLensVL adds the multimodal layer, the conflict/observer tooling, and the prompt helper.

---

## Why

Validated on **Qwen3.5-4B** (natively multimodal). Every output below is from a real run.

**1. A concept forms before it's spoken** — the true J-Lens is legible mid-stack where the plain logit-lens is noise:

```
prompt: "Fact: The currency used in the country shaped like a boot is the"
  L20  currency / called / country          logit-lens same layer: baku / 魄 / ernen  (noise)
  L25  euro / Euro / 欧元 / Euros
  L26  euro / Euro / Italian                 ← the latent "Italy" (boot = Italy) surfaces
  L30  Euro / euro / Italian
```

**2. The lens sees what a VLM doesn't say.** Feed a photo of a pug; the model answers just `"dog"`, but the J-Lens reveals it knew it was a **pug**:

```
pug.jpg   model says: 'dog'
  L25  dog / Dog / puppy
  L30  pug / Pug / dog          ← finer latent detail than the 1-word answer
dog.jpg (a black puppy)  →  L29-30: black + dog   (correct attribute, unsaid)
```

**3. Watch vision override a textual lie, layer by layer.** A dog photo, but the text claims it's a cat:

```
                  dog     cat
  L12–21          <       >     cat leads (prior/default)
  L22             >             ← VISION takes over
  L27          30.6    15.8     dog dominates (Δ +14.8)
Stronger textual lie ⇒ crossover delayed (L22→L24) and dog dominance halved. The conflict is quantifiable.
```

**4. Prompt helper — _see_ which phrasing is clearer.** Forward-only ranking of prompt variants by how strongly they steer the model to your intended sense:

```
Prompt-Helper report — intended sense: 'programming'
#1  [coding ctx]  "In software engineering, Java is a"
    programming  |██████████████████████| 21.12   ← intended
    island       |██████████            |  9.19
    → margin +11.94   [✓ CLEAR]
#3  [island ctx]  "On the map of Indonesia, Java is a"
    island       |███████████████████   | 18.50
    programming  |███████████           | 11.00   ← intended
    → margin  -7.50   [✗ OFF-TARGET]
VERDICT: use [coding ctx] — steers to 'programming' with +11.94 margin.
```

---

## Install

Needs a CUDA GPU (or MPS) with the model resident. For Qwen3.5's Gated-DeltaNet layers, **do not install `fla`/`causal-conv1d`** — the differentiable pure-PyTorch path is what makes the lens fittable.

```bash
pip install -e .          # pulls torch, transformers, pillow, torchvision, and the jacobian-lens engine
```

## Quickstart

```python
from jlensvl import JLensVL, PromptHelper

# load a VLM (vision tower auto-detected) with a fitted lens
jl = JLensVL.from_pretrained("Qwen/Qwen3.5-4B", lens="lens.pt")

# --- vision: what is the model poised to say about an image? ---
print(jl.describe("pug.jpg"))                       # -> 'dog'
print(jl.trace_image("pug.jpg", "What is this?")["answer"][30])   # -> ['pug', 'Pug', 'dog', ...]

# --- conflict: dog photo, text says cat ---
race = jl.concept_race("dog.jpg",
                       "This is a cat. What animal is this?",
                       {"dog": ["dog", "puppy"], "cat": ["cat", "kitten"]})

# --- prompt helper: rank phrasings, visually ---
ph = PromptHelper(jl)
print(ph.report(
    {"bare": "Java is a", "coding": "In software engineering, Java is a"},
    senses={"programming": ["programming", "language"], "island": ["island", "province"]},
    intended="programming"))
```

Don't have a lens yet? Fit one (does backward passes; ~15 min for a 4B model on a 24 GB GPU):

```bash
python scripts/fit_lens.py --model Qwen/Qwen3.5-4B --out lens.pt --n 100
```

See [`examples/`](examples/) for runnable text, vision, conflict, and prompt-helper demos.

## MLX backend (Apple Silicon, forward-only)

The Jacobian `J_ℓ` is fit **once, offline** (on CUDA or Apple MPS via torch); applying the
lens at inference is **forward-only** (`unembed(norm(J_ℓ · h))`). So on Apple Silicon you
don't need a custom Metal backward kernel for Qwen3.5's Gated-DeltaNet layers at all — you
just load the fitted `J_ℓ` into MLX and do a native MLX forward + a matmul:

```bash
# 1) fit J_ell offline (torch; do NOT install fla so GDN stays differentiable)
python scripts/fit_lens.py --model Qwen/Qwen3.5-4B --out lens.pt --n 100
# 2) export to a plain .npz MLX can read
python scripts/lens_to_npz.py lens.pt lens_jl.npz
# 3) run the native-MLX forward-only lens (loads mlx-community/Qwen3.5-4B-4bit)
python examples/05_mlx_forward_lens.py
```

Verified: the MLX forward-only lens reproduces the same `currency → euro → Italian`
readout as the torch lens — sub-second per forward, no `custom_gdn_vjp` Metal kernel.
Loading the MLX model sidesteps a `mlx_lm`/transformers-5.x clash by using
`mlx_lm.utils.load_model` + the raw `tokenizers` lib (see the example).

## API

| call | what |
|---|---|
| `JLensVL.from_pretrained(id, lens=...)` | load model (+processor) and a fitted lens; auto-detects a vision tower |
| `.fit(prompts)` / `.save_lens(path)` | fit `J_ℓ` on a corpus and save |
| `.trace(prompt)` | per-layer top-k J-Lens tokens for a text prompt |
| `.describe(image)` | the model's own short answer (behavioral reference) |
| `.trace_image(image, q)` | J-Lens at the answer, post-image, and image positions |
| `.concept_race(image, q, concepts)` | per-layer competition between concept sets |
| `PromptHelper.poised(prompt)` | what the prompt is poised to say + decisiveness margin |
| `PromptHelper.rank_prompts(variants, senses, intended)` | rank phrasings by intended-sense dominance |
| `PromptHelper.report(...)` | the same, as a visual ASCII-bar report |

## Caveats (honest)

- The lens reads *poised-to-say* content; it correlates with output but is not a perfect predictor. Use it as a **diagnostic**, with controls (baselines, aggregates), not a ground truth.
- The raw **logit-lens** is noisy mid-stack on these models — that's exactly why the fitted **Jacobian** lens exists.
- Needs **white-box** access (weights + a fitted lens). It won't work against a closed API.
- The prompt helper compares/diagnoses phrasings; it does not auto-generate prompts.

## Credits

- **J-Lens & the fitting engine:** [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) and the paper [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/) (Transformer Circuits, 2026).
- Prior multimodal exploration: [jerrickhoang/jlens-qwen-jspace](https://github.com/jerrickhoang/jlens-qwen-jspace); Apple-Silicon MLX port: [WeZZard/jlens-qwen36](https://github.com/WeZZard/jlens-qwen36).

JLensVL's contribution is the packaged **VLM** layer (multimodal readout, concept-race, prompt helper) on top of that engine.

## License

Apache-2.0. See [LICENSE](LICENSE).
