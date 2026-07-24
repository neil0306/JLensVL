"""Native MLX encoder + word-proposer query path for JLensVL (Apple Silicon).

The retrieval word-proposer (`RetrievalLens.propose_concept`) normally captures a
model's per-layer residual-stream hidden states through a torch
`ActivationRecorder` (`retrieval_lens._capture`). On Apple Silicon there is no
torch model — the model is an MLX-quantized Qwen3.5. This module provides the
**same capture seam natively in MLX**, so:

* the retrieval INDEX can be built from MLX hidden states (`MLXEncoder.encode_fn`,
  fed to `retrieval_lens.build_index(encode_fn=...)`), and
* the QUERY path (`propose_concept`) can read the true last-position hidden state
  from MLX and search the index directly (`propose_concept_mlx`) — mirroring the
  torch `_capture` -> `index.search` flow exactly.

`MLXEncoder` replays the ``Qwen3_5TextModel`` residual loop (embed_tokens -> each
DecoderLayer block output) and captures the selected block-output layers, matching
``retrieval_lens._ModelEncoder`` (index) and ``RetrievalLens._capture`` (query).

MLX / mlx_lm are imported **lazily** (inside methods), so this module imports fine
on a non-Apple machine that has torch but no MLX. numpy + torch are required (the
index tensors are torch tensors, on both platforms).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

from .retrieval_lens import (
    RetrievalLens,
    _has_cjk,
    _norm_word,
    default_layers,
)

#: matches ``retrieval_lens._ModelEncoder.CJK_GROUP_CAP`` — spaceless CJK runs
#: longer than this fall back to per-token surfaces instead of one giant "word".
CJK_GROUP_CAP = 4


def _word_map(hf, enc, ids):
    """Position -> surface word via the HF fast tokenizer's ``word_ids``
    (CJK-capped), mirroring ``retrieval_lens._ModelEncoder._word_map`` so MLX
    index metadata matches the torch build exactly."""
    try:
        word_ids = enc.word_ids(0)
    except (TypeError, ValueError, AttributeError):
        return None
    if word_ids is None:
        return None
    groups, positions = defaultdict(list), defaultdict(list)
    for pos, wid in enumerate(word_ids):
        if wid is not None:
            groups[wid].append(ids[pos])
            positions[wid].append(pos)
    word_of_pos = {}
    for wid, g in groups.items():
        surface = hf.decode(g).strip()
        if _has_cjk(surface) and len(surface) > CJK_GROUP_CAP:
            for pos in positions[wid]:
                word_of_pos[pos] = hf.decode([ids[pos]]).strip()
        else:
            for pos in positions[wid]:
                word_of_pos[pos] = surface
    return word_of_pos


class MLXEncoder:
    """Shared residual-stream capture over the MLX ``Qwen3_5TextModel``.

    ``capture(text)``  -> ``(ids, {L: Tensor[T, d]})`` for ALL positions — used by
        the query path, mirroring torch ``_capture`` which reads a true position.
    ``encode_fn(text)`` -> ``(metas, hiddens)`` for the index (skips BOS + pos 1,
        like LatentLens / ``_ModelEncoder``).

    Hidden states are returned as CPU float32 **torch** tensors so they drop
    straight into ``retrieval_lens.build_index`` / ``RetrievalIndex.search``.
    """

    def __init__(self, model, tokenizer, layers: Sequence[int],
                 max_length: int = 64) -> None:
        self.model = model
        self.tm = model.model            # Qwen3_5TextModel
        self.mlx_layers = self.tm.layers
        self.layers = list(layers)
        self.target = set(self.layers)
        self.hf = tokenizer._tokenizer   # underlying HF fast tokenizer
        self.max_length = max_length

    @classmethod
    def from_pretrained(cls, model_id: str,
                        layers: Optional[Sequence[int]] = None,
                        max_length: int = 64) -> "MLXEncoder":
        """Load an MLX model (``model_id`` = mlx-community id or local dir) and
        build an encoder. ``layers`` defaults to ``default_layers(depth)``."""
        from mlx_lm import load  # lazy: MLX only exists on Apple Silicon
        model, tokenizer = load(model_id)
        if layers is None:
            n = len(model.model.layers)
            layers = default_layers(n)
        return cls(model, tokenizer, layers, max_length)

    def _forward(self, ids):
        """Replay the residual loop and capture the target block outputs.
        Returns ``{L: torch.FloatTensor[T, d]}`` on CPU."""
        import mlx.core as mx
        import numpy as np
        import torch
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        x = mx.array([ids])
        h = self.tm.embed_tokens(x)
        fa_mask = create_attention_mask(h, None)
        ssm_mask = create_ssm_mask(h, None)
        captured = {}
        for i, layer in enumerate(self.mlx_layers):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=None)
            if i in self.target:
                captured[i] = h[0]       # [T, d]
        mx.eval(list(captured.values()))
        return {L: torch.from_numpy(np.array(captured[L].astype(mx.float32)))
                for L in self.layers}

    def capture(self, text: str):
        """Encode ``text`` and return ``(ids, {L: Tensor[T, d]})`` for every
        position (no skip) — the query path reads the true last position."""
        enc = self.hf(text, truncation=True, max_length=self.max_length)
        ids = enc["input_ids"]
        return ids, self._forward(ids)

    def encode_fn(self, text: str):
        """``text -> (metas, hiddens)`` for ``build_index``. Skips BOS + pos 1
        (``range(2, seq)``) and expands subword tokens to surface words, matching
        ``retrieval_lens._ModelEncoder``."""
        import torch
        enc = self.hf(text, truncation=True, max_length=self.max_length)
        ids = enc["input_ids"]
        word_of_pos = _word_map(self.hf, enc, ids)
        acts = self._forward(ids)
        seq = len(ids)
        rows, metas = [], []
        for pos in range(2, seq):
            tid = ids[pos]
            ts = self.hf.decode([tid])
            if word_of_pos is not None and pos in word_of_pos:
                word = word_of_pos[pos] or ts.strip()
            else:
                word = ts.strip()
            metas.append({"token_str": ts, "word": word, "position": pos,
                          "token_id": tid})
            rows.append(pos)
        idx = torch.tensor(rows, dtype=torch.long)
        hiddens = {L: acts[L][idx] for L in self.layers}
        return metas, hiddens


def propose_concept_mlx(index, enc: MLXEncoder, concept: str, k: int = 8,
                        template: str = "{}",
                        layers: Optional[Sequence[int]] = None) -> list:
    """MLX equivalent of ``RetrievalLens.propose_concept``.

    Reads the TRUE last-position hidden state of ``template.format(concept)`` from
    the MLX model (no BOS/pos-1 skip, mirroring torch ``_capture``), searches
    ``index`` per layer, excludes the concept + trailing token (self-echo), widens
    the fetch until ``k`` survive, and aggregates into ranked unique words. Return
    shape matches ``propose_concept`` (list of ``{word, score, ...}`` dicts).
    """
    text = template.format(concept)
    ids, full = enc.capture(text)              # full per-position residuals
    layers = list(layers) if layers is not None else index.available_layers
    q_by = {L: full[L][-1] for L in layers}    # true last position, like _capture
    last_tok = enc.hf.decode([ids[-1]])
    ex = {_norm_word(concept), _norm_word(last_tok)}
    cap = max(sum(index._layers_data[L]["vectors"].shape[0] for L in layers), 1)
    fetch = min(max(k * 8, k), cap)
    while True:
        neigh = index.search(q_by, k=fetch, layers=layers)[0]
        neigh = [n for n in neigh if _norm_word(n.word) not in ex
                 and _norm_word(n.token_str) not in ex]
        result = RetrievalLens._aggregate(neigh, k)
        if len(result) >= k or fetch >= cap:
            return result
        fetch = min(fetch * 2, cap)
