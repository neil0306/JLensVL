"""JLensVL — a Jacobian-Lens (J-Lens) observer for vision-language models.

Built on top of Anthropic's reference `jacobian-lens` engine (which fits the
per-layer average Jacobian `J_l`), JLensVL adds the multimodal layer: read what
a VLM is *poised to say* at any token position -- including image-token and
post-image text positions -- plus concept-competition ("race") analysis and a
forward-only prompt helper.

The lens itself is forward-only at inference: the `J_l` matrices are fitted once
(`JLensVL.fit`), then every readout is a single forward pass + a matmul.
"""
from __future__ import annotations
from typing import Sequence
import torch
from PIL import Image

import jlens
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder


class JLensVL:
    """A fitted J-Lens bound to a (multimodal) HuggingFace model.

    Attributes:
        model: the HF model (CausalLM or ImageTextToText).
        processor: AutoProcessor (VLM) or AutoTokenizer (text-only).
        tok: the tokenizer.
        lm: the `jlens` LensModel wrapper around the text decoder.
        lens: the fitted `JacobianLens` (or None until `.fit`).
        image_token_id: id of the image placeholder token (VLM), else None.
    """

    def __init__(self, model, processor, lm, lens=None):
        self.model = model
        self.processor = processor
        self.tok = getattr(processor, "tokenizer", processor)
        self.lm = lm
        self.lens = lens
        self.n_layers = lm.n_layers
        self.d_model = lm.d_model
        self.image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)

    # ---------- construction ----------
    @classmethod
    def from_pretrained(cls, model_id, *, lens=None, dtype=torch.bfloat16,
                        device="auto", multimodal="auto"):
        """Load `model_id` and wrap it. `lens` may be a path to a saved lens,
        a `JacobianLens`, or None (fit later). `multimodal="auto"` detects a
        vision tower from the config. `device="auto"` picks cuda if available,
        else mps (Apple Silicon), else cpu; an explicit "cuda"/"cuda:N"/"mps"/
        "cpu" is honored as given."""
        from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                                   AutoModelForCausalLM, AutoModelForImageTextToText)
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        use_cuda = device.startswith("cuda")
        cfg = AutoConfig.from_pretrained(model_id)
        is_mm = (multimodal is True) or (multimodal == "auto" and hasattr(cfg, "vision_config"))
        if is_mm:
            processor = AutoProcessor.from_pretrained(model_id)
            if use_cuda:
                model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map={"": device})
            else:
                model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device)
        else:
            processor = AutoTokenizer.from_pretrained(model_id)
            if use_cuda:
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map={"": device})
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        model.eval()
        tok = getattr(processor, "tokenizer", processor)
        lm = jlens.from_hf(model, tok)
        if isinstance(lens, str):
            lens = JacobianLens.from_pretrained(lens)
        return cls(model, processor, lm, lens)

    def fit(self, prompts: Sequence[str], **kw) -> JacobianLens:
        """Fit the lens on `prompts` (see `jlens.fit`). Stores and returns it."""
        self.lens = jlens.fit(self.lm, prompts, **kw)
        return self.lens

    def save_lens(self, path: str):
        self.lens.save(path)

    # ---------- low-level readout ----------
    def _require_lens(self):
        if self.lens is None:
            raise RuntimeError("no lens: call .fit(prompts) or pass lens= to from_pretrained")

    def _decode(self, ids):
        return [self.tok.decode([int(i)]).replace("\n", "\\n") for i in ids]

    def _readout_vec(self, residual, layer, k=8, use_jacobian=True):
        r = residual.float()
        if use_jacobian:
            self._require_lens()
            r = self.lens.transport(r, layer)
        logits = self.lm.unembed(r[None].to(residual.device))[0]
        vals, idx = logits.topk(k)
        return self._decode(idx.tolist()), vals

    def _capture(self, inputs, layers):
        with torch.no_grad(), ActivationRecorder(self.lm.layers, at=list(layers)) as rec:
            self.model(**inputs)
            return {i: rec.activations[i].detach() for i in layers}

    def _word_ids(self, words):
        """Single-token vocab ids for each word (with/without leading space, cased)."""
        s = set()
        for w in words:
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                e = self.tok.encode(v, add_special_tokens=False)
                if len(e) == 1:
                    s.add(e[0])
        return sorted(s)

    # ---------- text ----------
    def trace(self, prompt, *, layers=None, position=-1, k=8, use_jacobian=True):
        """Per-layer top-k J-Lens tokens at `position` for a text prompt.
        Returns {layer: [tokens]}."""
        self._require_lens()
        layers = list(layers) if layers is not None else self.lens.source_layers
        ll, _, _ = self.lens.apply(self.lm, prompt, positions=[position], layers=layers,
                                   use_jacobian=use_jacobian)
        return {L: self._decode(ll[L][0].topk(k).indices.tolist()) for L in sorted(ll)}

    # ---------- vision ----------
    def _vlm_inputs(self, image, question, enable_thinking=False):
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": question}]}]
        try:
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        except TypeError:
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)

    def describe(self, image, question="What is the main subject of this photo? Answer briefly.",
                 max_new_tokens=16):
        """The model's own short answer about `image` (behavioral reference)."""
        inputs = self._vlm_inputs(image, question)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def trace_image(self, image, question, *, layers=None, k=8):
        """Read the J-Lens on a VLM at three loci: the answer position, the first
        text position after the image, and the image-token span. Returns a dict."""
        self._require_lens()
        inputs = self._vlm_inputs(image, question)
        ids = inputs["input_ids"][0]
        img_pos = (ids == self.image_token_id).nonzero(as_tuple=True)[0].tolist() if self.image_token_id else []
        layers = list(layers) if layers is not None else self.lens.source_layers
        acts = self._capture(inputs, layers)
        answer = len(ids) - 1
        out = {"answer": {L: self._readout_vec(acts[L][0, answer], L, k)[0] for L in layers},
               "image_positions": img_pos}
        if img_pos:
            out["post_image"] = {L: self._readout_vec(acts[L][0, img_pos[-1] + 1], L, k)[0] for L in layers}
        return out

    def concept_race(self, image, question, concepts, *, layers=None, position="answer"):
        """Track competing concepts across layers (e.g. contradictory image/text).
        `concepts` = {name: [words]}. Returns {layer: {name: score}}."""
        self._require_lens()
        inputs = self._vlm_inputs(image, question)
        ids = inputs["input_ids"][0]
        pos = (len(ids) - 1) if position == "answer" else int(position)
        layers = list(layers) if layers is not None else self.lens.source_layers
        acts = self._capture(inputs, layers)
        cid = {n: self._word_ids(w) for n, w in concepts.items()}
        rows = {}
        for L in layers:
            r = acts[L][0, pos].float()
            logits = self.lm.unembed(self.lens.transport(r, L)[None].to(r.device))[0]
            rows[L] = {n: float(logits[i].max()) for n, i in cid.items()}
        return rows
