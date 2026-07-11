"""MLX forward-only backend for JLensVL (Apple Silicon).

The Jacobian `J_l` is fit ONCE offline (torch, on CUDA or MPS); applying the lens
is forward-only. So on Apple Silicon there is NO need for a custom Metal backward
kernel through Qwen3.5's Gated-DeltaNet layers — you just load the torch-fitted
`J_l` into MLX and do a native MLX forward + a matmul + unembed.

Workflow:
    # offline (torch): fit + export
    #   python scripts/fit_lens.py --model Qwen/Qwen3.5-4B --out lens.pt
    #   python scripts/lens_to_npz.py lens.pt lens_jl.npz
    # on the Mac (MLX):
    from jlensvl.mlx_backend import MLXJLens
    jl = MLXJLens.from_pretrained("mlx-community/Qwen3.5-4B-4bit", "lens_jl.npz")
    print(jl.trace("... the country shaped like a boot is the")[30])   # -> euro / Italian
    jl.slice_grid_html("...", out_path="slice.html")

Needs the [mlx] extra: `pip install mlx mlx-lm tokenizers`. MLX + mlx_lm are
imported lazily so `import jlensvl` still works on non-Apple machines.

Note: loading the MLX model uses `mlx_lm.utils.load_model` + the raw `tokenizers`
lib to sidestep an mlx_lm/transformers-5.x tokenizer-class clash.
"""
from __future__ import annotations
import glob
import os


def _resolve_snapshot(model_id_or_path: str) -> str:
    """Return a local snapshot dir for an MLX model id (or pass through a path)."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    key = model_id_or_path.replace("/", "--")
    hits = sorted(glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{key}/snapshots/*")))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"no local snapshot for {model_id_or_path!r}; download it first "
        f"(e.g. `huggingface-cli download {model_id_or_path}`) or pass a local dir")


class _Cap:
    """Wraps an MLX decoder layer to record its residual output during forward."""
    def __init__(self, layer, idx, store):
        self.layer, self.idx, self.store = layer, idx, store

    def __call__(self, *a, **k):
        out = self.layer(*a, **k)
        self.store[self.idx] = out[0] if isinstance(out, (tuple, list)) else out
        return out

    def __getattr__(self, n):
        return getattr(object.__getattribute__(self, "layer"), n)


class MLXJLens:
    """Forward-only J-Lens on an MLX-quantized model, using a torch-fitted J_l."""

    def __init__(self, model, tokenizer, J, decoder, norm, embed):
        self.model = model
        self.tok = tokenizer
        self.J = J
        self.layers = sorted(J.keys())
        self.decoder = decoder
        self.norm = norm
        self.embed = embed

    @classmethod
    def from_pretrained(cls, mlx_model, lens_npz, *, model_snapshot=None):
        """Load an MLX model (`mlx_model` = an mlx-community id or a local dir) and
        the exported lens (`lens_npz` from scripts/lens_to_npz.py)."""
        import mlx.core as mx
        import numpy as np
        from pathlib import Path
        from mlx_lm.utils import load_model
        from tokenizers import Tokenizer

        snap = model_snapshot or _resolve_snapshot(mlx_model)
        model, _ = load_model(Path(snap))
        tk = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
        lm = getattr(model, "language_model", None) or model
        inner = getattr(lm, "model", lm)
        decoder, norm, embed = inner.layers, inner.norm, inner.embed_tokens
        z = np.load(lens_npz)
        J = {int(k): mx.array(z[k].astype(np.float32)) for k in z.files}
        return cls(model, tk, J, decoder, norm, embed)

    # ---- internals ----
    def _forward_capture(self, ids):
        import mlx.core as mx
        acts = {}
        originals = {}
        for i in self.layers:
            originals[i] = self.decoder[i]
            self.decoder[i] = _Cap(originals[i], i, acts)
        try:
            logits = self.model(mx.array([ids]))
            mx.eval(logits, *acts.values())
        finally:
            for i, o in originals.items():
                self.decoder[i] = o
        return acts, logits

    def _readout(self, h, L, k, use_jacobian=True):
        import mlx.core as mx
        t = (h @ self.J[L].T) if use_jacobian else h
        lg = self.embed.as_linear(self.norm(t[None]))[0]
        idx = [int(i) for i in mx.argsort(-lg)[:k].tolist()]
        return [self.tok.decode([i]).replace("\n", "\\n") for i in idx], lg

    # ---- public ----
    def trace(self, prompt, *, layers=None, position=-1, k=8, use_jacobian=True):
        """Per-layer top-k J-Lens tokens at `position` for a text prompt.
        Returns {layer: [tokens]}."""
        import mlx.core as mx
        ids = self.tok.encode(prompt).ids
        acts, _ = self._forward_capture(ids)
        layers = sorted(layers) if layers is not None else self.layers
        out = {}
        for L in layers:
            h = acts[L][0, position].astype(mx.float32)
            out[L] = self._readout(h, L, k, use_jacobian)[0]
        return out

    def slice_grid_html(self, prompt, *, layers=None, topk=5, out_path=None,
                        title="JLensVL MLX slice grid"):
        """Layer x position slice grid (self-contained HTML) computed natively in MLX."""
        import mlx.core as mx
        import html as _h
        from . import viz
        ids = self.tok.encode(prompt).ids
        acts, _ = self._forward_capture(ids)
        layers = sorted(layers) if layers is not None else self.layers
        toks = [(self.tok.decode([i]).replace("\n", "\\n").strip() or "·") for i in ids]
        seq = len(ids)
        cell, vmax = {}, 1e-6
        for L in layers:
            for p in range(seq):
                h = acts[L][0, p].astype(mx.float32)
                top, lg = self._readout(h, L, topk)
                s = float(mx.max(lg))
                vmax = max(vmax, s)
                cell[(L, p)] = ([t.strip() or "·" for t in top], s)

        def color(s):
            a = max(0.0, min(1.0, s / vmax))
            return f"background:rgba(126,231,135,{a*0.8:.2f})"

        head = "".join(f'<th title="position {p}">{_h.escape(toks[p])}</th>' for p in range(seq))
        rows = ""
        for L in reversed(layers):
            cells = "".join(
                f'<td style="{color(cell[(L,p)][1])}" title="{_h.escape(" · ".join(cell[(L,p)][0]))}">'
                f'{_h.escape(cell[(L,p)][0][0])}</td>' for p in range(seq))
            rows += f'<tr><th class="ly">L{L}</th>{cells}</tr>'
        doc = (viz._HEAD.format(title=_h.escape(title)) +
               f'<h1>{_h.escape(title)}</h1>'
               f'<p class="sub">MLX 前向式 J-Lens（无 Metal 反向 kernel）· 越绿=越确定 · hover 看 top-{topk} · 深层在上</p>'
               f'<div class="wrap"><table><thead><tr><th class="ly">layer</th>{head}</tr></thead>'
               f'<tbody>{rows}</tbody></table></div>'
               f'<p class="legend">prompt: {_h.escape(prompt)}</p></body></html>')
        if out_path:
            open(out_path, "w").write(doc)
        return doc
