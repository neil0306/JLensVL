"""MLX forward-only J-Lens (Apple Silicon). Native MLX forward + transport by the
torch-fitted J_ell (exported to .npz) + unembed. No custom Metal backward kernel.

Run in an MLX env:  pip install mlx mlx-lm tokenizers
  MLX_MODEL=mlx-community/Qwen3.5-4B-4bit  LENS_NPZ=lens_jl.npz  python examples/05_mlx_forward_lens.py

Note: loads the model via mlx_lm.utils.load_model + the raw `tokenizers` lib to
sidestep an mlx_lm/transformers-5.x tokenizer clash.
"""
import glob, os, numpy as np
import mlx.core as mx
from mlx_lm.utils import load_model
from tokenizers import Tokenizer
from pathlib import Path

MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen3.5-4B-4bit")
LENS_NPZ = os.environ.get("LENS_NPZ", "lens_jl.npz")
PROBE = os.environ.get("PROBE", "Fact: The currency used in the country shaped like a boot is the")

# resolve a local snapshot dir for the MLX model
cand = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/','--')}/snapshots/*"))
snap = sorted(cand)[0] if cand else MODEL
model, config = load_model(Path(snap))
tk = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))

lm = getattr(model, "language_model", None) or model      # text model wraps as language_model.model
inner = getattr(lm, "model", lm)
decoder, norm, embed = inner.layers, inner.norm, inner.embed_tokens

z = np.load(LENS_NPZ)
J = {int(k): mx.array(z[k].astype(np.float32)) for k in z.files}
LAYERS = sorted(J.keys())

# capture residuals by wrapping the decoder layers
acts = {}
class _Cap:
    def __init__(self, layer, idx): self.layer, self.idx = layer, idx
    def __call__(self, *a, **k):
        out = self.layer(*a, **k)
        acts[self.idx] = out[0] if isinstance(out, (tuple, list)) else out
        return out
    def __getattr__(self, n): return getattr(object.__getattribute__(self, "layer"), n)
for i in LAYERS:
    decoder[i] = _Cap(decoder[i], i)

ids = tk.encode(PROBE).ids
logits = model(mx.array([ids])); mx.eval(logits)
print("model argmax @last:", repr(tk.decode([int(mx.argmax(logits[0, -1]))])))

def readout(L, pos=-1, k=6, use_jacobian=True):
    h = acts[L][0, pos].astype(mx.float32)
    t = (h @ J[L].T) if use_jacobian else h
    lg = embed.as_linear(norm(t[None]))[0]
    return [tk.decode([int(i)]).replace("\n", "\\n") for i in mx.argsort(-lg)[:k].tolist()]

print("\nMLX J-Lens (forward-only, torch-fitted J_ell) @ last position:")
for L in [20, 24, 25, 26, 27, 29, 30]:
    if L in acts:
        print(f"  L{L:02d} {readout(L)}")
