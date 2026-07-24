"""Retrieval-based lens: a *word proposer* built on the model's own workspace.

Where the J-Lens reads what a model is *poised to say* through its unembedding,
the retrieval lens reads a hidden state and returns the words/phrases **this
model natively associates with it**, retrieved from an index of the model's own
hidden states over a text corpus.

Reimplementation of McGill-NLP LatentLens (`latentlens/index.py`,
`latentlens/extract.py`) for a decoder-only LM, with two deliberate choices:

* The index vectors are the **probed model's own** per-token hidden states, so
  query space == index space — no alignment or trained connector is needed
  (query and index just have to be captured the same way; we use one
  `ActivationRecorder` path for both).
* We keep a **hand-picked layer subset** and do brute-force cosine search
  (per-layer top-k + a global cross-layer re-rank), no FAISS.

Everything here is model-agnostic: `default_layers` is derived from the model's
depth, and the only Qwen3-specific fact (bf16, since fp16 overflows in early
layers) lives in how the *model* is loaded, not here — captured hidden states
are up-cast to float32 for the index regardless.
"""

from __future__ import annotations

import string
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

# Subword / boundary markers various tokenizers prepend (SentencePiece "▁",
# GPT-2 byte-BPE "Ġ"/"Ċ") — stripped when comparing a neighbor to the query so
# "▁cigarette" / "-cigarette" / "/vape" don't read as self-echoes.
_MARKERS = "▁ĠċĠĊ"
# Only *wrapping* punctuation is stripped from the edges — quotes, brackets,
# slashes, dashes, commas, periods and the like. Word-internal / trailing
# symbols that are part of the identity ("C++", "C#", "F#") are preserved, so a
# token like "C" is not treated as an echo of "C++".
_WRAP_PUNCT = "\"'`“”‘’()[]{}<>/\\-–—.,;:!?*|~@ "
_STRIP = _WRAP_PUNCT + string.whitespace + _MARKERS


def _norm_word(w: str) -> str:
    """Normalize a token/word for self-echo comparison: lowercase, drop subword
    markers, strip *wrapping* punctuation/whitespace from the edges. Preserves
    word-internal/trailing identity symbols like the ``++`` in ``C++``."""
    if not w:
        return ""
    w = w.lower()
    for m in _MARKERS:
        w = w.replace(m, "")
    return w.strip(_STRIP)


#: Codepoint ranges for scripts written without inter-word spaces. A fast
#: tokenizer's whitespace-based pre-tokenization can lump an entire spaceless
#: run (up to a whole sentence) under one ``word_id`` — which would otherwise
#: surface as a sentence-length "word". Covers CJK ideographs (BMP + Ext A/B and
#: compatibility), kana (incl. halfwidth), Bopomofo and Hangul (syllables + jamo).
_CJK_RANGES = (
    (0x1100, 0x11FF),    # Hangul Jamo
    (0x3040, 0x30FF),    # Hiragana + Katakana
    (0x3100, 0x312F),    # Bopomofo
    (0x3400, 0x9FFF),    # CJK Unified Ideographs (+ Ext A)
    (0xAC00, 0xD7A3),    # Hangul syllables
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0xFF66, 0xFF9D),    # Halfwidth Katakana
    (0x20000, 0x2FFFF),  # CJK Unified Ideographs Ext B..F (astral plane)
)


def _has_cjk(s: str) -> bool:
    """True if `s` contains any CJK / Japanese / Korean character (see
    ``_CJK_RANGES``)."""
    return any(
        any(lo <= cp <= hi for lo, hi in _CJK_RANGES)
        for cp in map(ord, s)
    )


# ── layer subset ──────────────────────────────────────────────────────────
def default_layers(n_layers: int) -> list[int]:
    """A sensible early/mid/late layer subset for a model of depth `n_layers`.

    Re-derives the LatentLens heuristic ``[1,2,4,8,16,24,n-2,n-1]`` in
    **block-output index space** (0..n_layers-1 — the indexing an
    `ActivationRecorder` over the residual blocks produces). For Qwen3.5-4B
    (32 layers) this is ``[1, 2, 4, 8, 16, 24, 30, 31]``.
    """
    base = [l for l in (1, 2, 4, 8, 16, 24) if 0 < l < n_layers - 2]
    top = [n_layers - 2, n_layers - 1]
    return sorted(set(base + [l for l in top if l >= 0]))


# ── result type ───────────────────────────────────────────────────────────
@dataclass
class Neighbor:
    """One nearest-neighbor hit from the index."""

    word: str            # surface word the token belongs to (subword-expanded)
    token_str: str       # the raw token string
    source_sentence: str # the corpus sentence it came from
    source_tag: str      # provenance tag, e.g. "vg" or "scenario:general"
    position: int        # token position in the source sentence
    token_id: int
    score: float         # cosine similarity to the query
    layer: int           # which index layer produced the hit


# ── index ─────────────────────────────────────────────────────────────────
class RetrievalIndex:
    """Per-layer store of L2-normalized hidden-state vectors + token metadata.

    `layers_data` maps a layer index to
    ``{"vectors": Tensor[N, d] (unit-norm), "meta": list[dict]}`` where each
    metadata row carries ``token_str, word, position, token_id,
    source_sentence, source_tag``.
    """

    def __init__(self, layers_data: dict[int, dict],
                 dtype: torch.dtype = torch.float32) -> None:
        self._layers_data = layers_data
        self.dtype = dtype

    # -- properties --
    @property
    def available_layers(self) -> list[int]:
        return sorted(self._layers_data)

    @property
    def hidden_dim(self) -> int:
        for ld in self._layers_data.values():
            return ld["vectors"].shape[1]
        raise ValueError("empty index")

    def __len__(self) -> int:
        return sum(ld["vectors"].shape[0] for ld in self._layers_data.values())

    def __repr__(self) -> str:
        ls = self.available_layers
        return (f"RetrievalIndex(layers={ls}, entries={len(self):,}, "
                f"dim={self.hidden_dim}, dtype={self.dtype})")

    def to(self, device: Union[str, torch.device]) -> "RetrievalIndex":
        """Move every layer's vectors to `device`, **in place, once**. Call this
        once before a batch of searches (e.g. ``index.to("cuda")``); `search`
        then runs on whatever device the vectors already live on and never
        copies the full index per call."""
        for ld in self._layers_data.values():
            ld["vectors"] = ld["vectors"].to(device)
        return self

    # -- search --
    @staticmethod
    def _chunked_topk(q: torch.Tensor, emb: torch.Tensor, k: int,
                      chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k cosine sims of `q` [T,d] against `emb` [N,d] without ever
        materializing the full [T,N] score matrix — bounds peak memory so a
        large index does not OOM the GPU."""
        n = emb.shape[0]
        best_v = best_i = None
        for s in range(0, n, chunk):
            e = emb[s:s + chunk]
            sim = q @ e.T                                 # [T, <=chunk]
            v, i = sim.topk(min(k, e.shape[0]), dim=-1)
            i = i + s
            if best_v is None:
                best_v, best_i = v, i
            else:
                v = torch.cat([best_v, v], dim=-1)
                i = torch.cat([best_i, i], dim=-1)
                v, sel = v.topk(min(k, v.shape[1]), dim=-1)
                i = torch.gather(i, 1, sel)
                best_v, best_i = v, i
        return best_v, best_i

    def search(self, query, k: int = 8,
               layers: Optional[Sequence[int]] = None,
               chunk: int = 200_000) -> list[list[Neighbor]]:
        """Top-k neighbors per query vector, globally re-ranked across layers.

        `query` is a unit-or-raw Tensor ``[d]`` / ``[T, d]`` (same vector used
        against every searched layer, as in LatentLens), OR a
        ``dict{layer: Tensor}`` to query each index layer with its *own* layer's
        hidden state (what `propose` does). Returns ``results[t]`` = list of up
        to `k` `Neighbor`s for query token `t`, best-first.

        Runs on the device the index vectors already live on (see `to`); the
        (small) query is moved/cast to match, so the (large) index is never
        copied. `chunk` bounds the score-matrix memory per matmul.
        """
        layers = sorted(layers) if layers is not None else self.available_layers

        def as2d(q):
            q = q.float()  # normalize in fp32 for stability
            return q.unsqueeze(0) if q.dim() == 1 else q

        if isinstance(query, dict):
            q_by = {L: F.normalize(as2d(query[L]), dim=-1) for L in layers}
        else:
            q2 = F.normalize(as2d(query), dim=-1)
            q_by = {L: q2 for L in layers}
        n_tokens = next(iter(q_by.values())).shape[0]

        per_layer = []  # (vals[T,k], idxs[T,k], layer)
        for L in layers:
            emb = self._layers_data[L]["vectors"]        # stays where it is
            # move/cast the small query to the index (not the reverse)
            q = q_by[L].to(device=emb.device, dtype=emb.dtype)
            vals, idxs = self._chunked_topk(q, emb, k, chunk)
            per_layer.append((vals.float().cpu(), idxs.cpu(), L))

        results: list[list[Neighbor]] = []
        for t in range(n_tokens):
            flat = []  # (score, layer, emb_idx)
            for vals, idxs, L in per_layer:
                for j in range(vals.shape[1]):
                    flat.append((vals[t, j].item(), L, int(idxs[t, j].item())))
            flat.sort(key=lambda x: -x[0])
            neigh = []
            for score, L, ei in flat[:k]:
                m = self._layers_data[L]["meta"][ei]
                neigh.append(Neighbor(
                    word=m.get("word") or m["token_str"].strip(),
                    token_str=m["token_str"], source_sentence=m["source_sentence"],
                    source_tag=m["source_tag"], position=m["position"],
                    token_id=m["token_id"], score=score, layer=L))
            results.append(neigh)
        return results

    # -- I/O (single-file .pt) --
    #: bumped when the on-disk layout changes; `load` refuses other versions.
    SCHEMA_VERSION = 1

    def save(self, path: Union[str, Path]) -> None:
        """Save to a single ``.pt``. The blob is only tensors + plain
        dict/list/str/int, so it reloads under ``weights_only=True`` (no pickle
        RCE surface). Layer keys are stringified for that safe path."""
        torch.save({"version": self.SCHEMA_VERSION, "dtype": str(self.dtype),
                    "layers_data": {str(L): {"vectors": ld["vectors"].cpu(),
                                             "meta": ld["meta"]}
                                    for L, ld in self._layers_data.items()}},
                   str(path))

    @classmethod
    def load(cls, path: Union[str, Path],
             map_location: Union[str, torch.device] = "cpu") -> "RetrievalIndex":
        """Load with ``weights_only=True`` (safe unpickling) and validate the
        schema before trusting the contents."""
        blob = torch.load(str(path), map_location=map_location, weights_only=True)
        if not isinstance(blob, dict) or "layers_data" not in blob:
            raise ValueError(f"{path}: not a RetrievalIndex file")
        ver = blob.get("version")
        if ver != cls.SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported index schema version {ver!r} "
                f"(expected {cls.SCHEMA_VERSION})")
        layers_data: dict[int, dict] = {}
        for L, ld in blob["layers_data"].items():
            if "vectors" not in ld or "meta" not in ld:
                raise ValueError(f"{path}: layer {L} missing vectors/meta")
            layers_data[int(L)] = {"vectors": ld["vectors"], "meta": ld["meta"]}
        dtype = next(iter(layers_data.values()))["vectors"].dtype if layers_data \
            else torch.float32
        return cls(layers_data, dtype=dtype)


# ── model encoder (the only piece that touches the model) ─────────────────
class _ModelEncoder:
    """Callable ``text -> (metas, hiddens)`` producing per-token hidden states.

    Captures block-output residuals via `ActivationRecorder` over `jl.lm.layers`
    — the same path JLensVL uses — so index and query vectors live in the same
    space. Skips BOS + position 1 (matching LatentLens `range(2, len)`), and
    expands subword tokens to their surface word via the fast tokenizer's
    ``word_ids`` when available.
    """

    #: Max chars a grouped CJK surface may span before we stop grouping and fall
    #: back to per-token units. CJK has no word spaces, so word_ids can lump a
    #: whole sentence into one group; capping keeps proposals to clean short
    #: surface units (typical CJK words are 1-4 chars) instead of sentence-length
    #: fragments. English grouping is untouched (it is space-delimited already).
    CJK_GROUP_CAP = 4

    def __init__(self, jl, layers: Sequence[int], max_length: int = 64) -> None:
        self.jl = jl
        self.layers = list(layers)
        self.max_length = max_length

    def _word_map(self, enc, ids: list[int]) -> Optional[dict[int, str]]:
        """Map each token *position* -> its surface word via the fast tokenizer's
        ``word_ids``. Subword pieces of one word share a word id and decode
        together (clean English grouping, unchanged). A group whose decoded
        surface is CJK and longer than ``CJK_GROUP_CAP`` chars — i.e. a spaceless
        run the pre-tokenizer collapsed into one "word" — falls back to
        per-token surfaces, so proposals stay short words, not sentence
        fragments. Returns None when the tokenizer exposes no ``word_ids``."""
        try:
            word_ids = enc.word_ids(0)
        except (TypeError, ValueError, AttributeError):
            return None
        if word_ids is None:
            return None
        tok = self.jl.tok
        groups: dict[int, list[int]] = defaultdict(list)   # wid -> token ids
        positions: dict[int, list[int]] = defaultdict(list)  # wid -> positions
        for pos, wid in enumerate(word_ids):
            if wid is not None:
                groups[wid].append(ids[pos])
                positions[wid].append(pos)
        word_of_pos: dict[int, str] = {}
        for wid, g in groups.items():
            surface = tok.decode(g).strip()
            if _has_cjk(surface) and len(surface) > self.CJK_GROUP_CAP:
                # spaceless CJK run collapsed into one word_id -> per-token units
                for pos in positions[wid]:
                    word_of_pos[pos] = tok.decode([ids[pos]]).strip()
            else:
                for pos in positions[wid]:
                    word_of_pos[pos] = surface
        return word_of_pos

    def __call__(self, text: str):
        from jlens.hooks import ActivationRecorder  # lazy: keeps module import-safe

        tok = self.jl.tok
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=self.max_length)
        dev = self.jl.lm.input_device
        ids_t = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        ids = ids_t[0].tolist()

        word_of_pos = self._word_map(enc, ids)

        with torch.no_grad(), ActivationRecorder(self.jl.lm.layers, at=self.layers) as rec:
            if attn is not None:
                self.jl.model(input_ids=ids_t, attention_mask=attn.to(dev),
                              use_cache=False)
            else:
                self.jl.model(input_ids=ids_t, use_cache=False)
            # move to CPU immediately: the index is accumulated on CPU so a big
            # corpus never keeps N×d activations resident in GPU memory.
            acts = {L: rec.activations[L][0].detach().float().cpu()
                    for L in self.layers}

        seq = ids_t.shape[1]
        rows, metas = [], []
        for pos in range(2, seq):
            tid = ids[pos]
            ts = tok.decode([tid])
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


# ── build entry point ─────────────────────────────────────────────────────
def _normalize_corpus(corpus) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in corpus:
        if isinstance(item, (tuple, list)):
            text, tag = item[0], (item[1] if len(item) > 1 else "corpus")
        elif isinstance(item, dict):
            text, tag = item.get("text", ""), item.get("source", "corpus")
        else:
            text, tag = item, "corpus"
        text = (text or "").strip()
        if text:
            out.append((text, tag))
    return out


def build_index(jl, corpus, *, layers: Optional[Sequence[int]] = None,
                reservoir_cap: int = 40, dtype: torch.dtype = torch.float32,
                encode_fn: Optional[Callable] = None,
                show_progress: bool = True) -> RetrievalIndex:
    """Build a `RetrievalIndex` from a JLensVL instance + a tagged corpus.

    Args:
        jl: a `JLensVL` (or any object exposing ``.tok``, ``.model`` and a
            ``.lm`` with ``layers`` / ``n_layers`` / ``input_device``). Only
            used to construct the default `encode_fn`.
        corpus: iterable of ``str`` | ``(text, source_tag)`` | ``{"text","source"}``.
        layers: index layers (block-output indices). Defaults to
            `default_layers(jl.lm.n_layers)`.
        reservoir_cap: soft first-come cap on entries per unique token string
            (bounds index size; LatentLens uses the same first-come cap).
        dtype: storage dtype for the (already L2-normalized) vectors.
        encode_fn: ``text -> (metas, hiddens)`` seam; defaults to `_ModelEncoder`.
            Override it (e.g. with a random stub) to build without a model.
    """
    if layers is None:
        n = getattr(getattr(jl, "lm", None), "n_layers", None)
        if n is None:
            raise ValueError("pass layers= (could not infer model depth)")
        layers = default_layers(n)
    layers = list(layers)
    corpus = _normalize_corpus(corpus)
    if encode_fn is None:
        encode_fn = _ModelEncoder(jl, layers)

    iterator = corpus
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(corpus, desc="Building retrieval index", unit="sent")
        except ImportError:
            pass

    layer_vecs: dict[int, list[torch.Tensor]] = defaultdict(list)
    layer_meta: dict[int, list[dict]] = defaultdict(list)
    token_counts: dict[str, int] = defaultdict(int)

    for text, tag in iterator:
        metas, hiddens = encode_fn(text)
        for i, m in enumerate(metas):
            ts = m["token_str"]
            if token_counts[ts] >= reservoir_cap:
                continue
            token_counts[ts] += 1
            meta = {**m, "source_sentence": text, "source_tag": tag}
            for L in layers:
                # .cpu() defensively: the index is accumulated/stored on CPU so
                # a large corpus never pins N×d vectors in GPU memory.
                layer_vecs[L].append(hiddens[L][i].detach().cpu())
                layer_meta[L].append(meta)

    layers_data: dict[int, dict] = {}
    for L in layers:
        if not layer_vecs[L]:
            continue
        V = F.normalize(torch.stack(layer_vecs[L]).float(), dim=-1).to(dtype)
        layers_data[L] = {"vectors": V, "meta": layer_meta[L]}
    return RetrievalIndex(layers_data, dtype=dtype)


# ── reader ────────────────────────────────────────────────────────────────
class RetrievalLens:
    """Reads hidden states against a `RetrievalIndex` to propose model-native
    words. Consumed directly by a human — no external judge."""

    def __init__(self, index: RetrievalIndex) -> None:
        self.index = index

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RetrievalLens":
        return cls(RetrievalIndex.load(path))

    def read(self, hidden: torch.Tensor, layer: Optional[int] = None,
             k: int = 8) -> list[Neighbor]:
        """Ranked neighbors for a single hidden vector `hidden`. If `layer` is
        given the query is matched only against that index layer; otherwise the
        same vector is matched across all layers and globally re-ranked."""
        if layer is None:
            return self.index.search(hidden, k=k)[0]
        return self.index.search({layer: hidden}, k=k, layers=[layer])[0]

    def _capture(self, jl, prompt: str, position: int,
                 layers: Sequence[int]) -> dict[int, torch.Tensor]:
        from jlens.hooks import ActivationRecorder
        ids = jl.lm.encode(prompt)
        seq = ids.shape[1]
        # normalize + bounds-check the position before indexing (a bad position
        # otherwise surfaces as a bare IndexError deep in the tensor index)
        idx = position + seq if position < 0 else position
        if not (0 <= idx < seq):
            raise ValueError(
                f"position {position} out of range for a {seq}-token sequence "
                f"(valid: {-seq}..{seq - 1})")
        with torch.no_grad(), ActivationRecorder(jl.lm.layers, at=list(layers)) as rec:
            jl.model(input_ids=ids, use_cache=False)
            acts = {L: rec.activations[L][0].detach().float() for L in layers}
        return {L: acts[L][idx] for L in layers}

    @staticmethod
    def _exclude_norms(exclude) -> set[str]:
        if exclude is None:
            return set()
        if isinstance(exclude, str):
            exclude = [exclude]
        return {n for n in (_norm_word(e) for e in exclude) if n}

    def _self_word(self, jl, prompt: str, position: int) -> Optional[str]:
        """Best-effort surface token at `position` of `prompt` (for self-echo
        filtering). Returns None if the tokenizer isn't available (e.g. tests)."""
        try:
            ids = jl.lm.encode(prompt)[0].tolist()
            return jl.tok.decode([ids[position]])
        except Exception:  # noqa: BLE001 - jl may be a stub / None
            return None

    def propose(self, jl, prompt: str, position: int = -1, k: int = 8,
                layers: Optional[Sequence[int]] = None,
                aggregate: bool = True, exclude_self: bool = True,
                exclude=None) -> list:
        """Model-native words at `position` of `prompt`.

        Captures the hidden state at each index layer for that position, queries
        each index layer with its own layer's vector, cross-layer re-ranks, then
        (default) aggregates neighbors into ranked unique words. Set
        ``aggregate=False`` to get the raw `Neighbor` list instead.

        `position` selects which token to read — pass an interior index (not just
        the default -1) to read a *content* position rather than a trailing
        determiner (e.g. reading "...holding a" at -1 mostly returns "a").

        Self-echo filtering (`exclude_self=True`, on by default): drops neighbors
        equal to the read token / any word in `exclude` up to case, surrounding
        punctuation and subword markers — so the proposer surfaces **alternatives
        and expansions** rather than confirming the word already there. Pass
        `exclude_self=False` to keep echoes. `exclude` adds extra words to drop.
        """
        layers = list(layers) if layers is not None else self.index.available_layers
        q_by = self._capture(jl, prompt, position, layers)
        ex = self._exclude_norms(exclude)
        if exclude_self:
            sw = self._self_word(jl, prompt, position)
            if sw:
                ex.add(_norm_word(sw))

        # Iteratively widen the over-fetch: if self-echo filtering leaves fewer
        # than k results, double the fetch and retry so we surface deeper
        # alternatives (rank > fetch) instead of returning a short/empty list.
        cap = sum(self.index._layers_data[L]["vectors"].shape[0] for L in layers)
        cap = max(cap, 1)
        fetch = min(max(k * 8, k), cap)
        while True:
            neigh = self.index.search(q_by, k=fetch, layers=layers)[0]
            if ex:
                neigh = [n for n in neigh
                         if _norm_word(n.word) not in ex
                         and _norm_word(n.token_str) not in ex]
            result = self._aggregate(neigh, k) if aggregate else neigh[:k]
            if len(result) >= k or fetch >= cap:
                return result
            fetch = min(fetch * 2, cap)

    def propose_concept(self, jl, concept: str, k: int = 8,
                        template: str = "{}",
                        layers: Optional[Sequence[int]] = None,
                        exclude_self: bool = True, exclude=None) -> list:
        """Propose model-native words for a named `concept` by reading the last
        position of ``template.format(concept)`` (e.g. template=\"The topic is {}\").

        With `exclude_self=True` (default) the concept word itself (and marker/
        punctuation variants like "▁cigarette", "cigarettes"→no, "/cigarette")
        is filtered out, so you get associations/alternatives instead of the
        concept echoing itself back."""
        ex = list(self._exclude_norms(exclude))
        if exclude_self:
            ex.append(concept)
        return self.propose(jl, template.format(concept), position=-1, k=k,
                            layers=layers, exclude_self=exclude_self, exclude=ex)

    def propose_rendered(self, jl, messages, position: int = -1, k: int = 8,
                         layers: Optional[Sequence[int]] = None,
                         enable_thinking: Optional[bool] = None,
                         add_generation_prompt: bool = True,
                         aggregate: bool = True, exclude_self: bool = True,
                         exclude=None) -> list:
        """Like `propose`, but reads the **real chat-template-rendered token
        sequence** for `messages` (what the model actually sees) instead of a raw
        string. Reuses `PromptHelper._render` (the same rendering path
        `PromptHelper.trace_rendered` uses), then retrieves on the hidden state at
        `position` of the rendered sequence.

        `messages` is the usual chat list ``[{"role","content"}, ...]``. Use an
        interior `position` to read a content token of the draft rather than the
        trailing generation-prompt tokens.
        """
        from .prompt_helper import PromptHelper
        if not getattr(getattr(jl, "tok", None), "chat_template", None):
            raise ValueError(
                "propose_rendered requires a chat template, but this "
                "tokenizer has none (tok.chat_template is empty). Use "
                "`propose` with a raw prompt string instead.")
        try:
            rendered = PromptHelper(jl)._render(
                messages, add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"propose_rendered failed to render the chat template: {exc}"
            ) from exc
        return self.propose(jl, rendered, position=position, k=k, layers=layers,
                            aggregate=aggregate, exclude_self=exclude_self,
                            exclude=exclude)

    @staticmethod
    def _aggregate(neighbors: list[Neighbor], k: int) -> list[dict]:
        """Collapse neighbors into ranked unique words (max score per word)."""
        best: dict[str, dict] = {}
        for n in neighbors:
            key = n.word.lower().strip()
            if not key:
                continue
            cur = best.get(key)
            if cur is None or n.score > cur["score"]:
                best[key] = {"word": n.word, "score": n.score,
                             "source_tag": n.source_tag,
                             "source_sentence": n.source_sentence,
                             "layer": n.layer, "count": 0}
            best[key]["count"] += 1
        ranked = sorted(best.values(), key=lambda d: -d["score"])
        return ranked[:k]
