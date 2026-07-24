"""CPU-only smoke test for the retrieval lens.

Builds a TINY index from ~10 fake sentences using a random-embedding stub for
the encode step (no GPU, no model load, no tokenizer), then exercises
build -> save -> load -> read/propose and checks the results are well-formed
ranked structures. This proves the whole code path with nothing but torch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from jlensvl.retrieval_lens import (
    Neighbor,
    RetrievalIndex,
    RetrievalLens,
    _ModelEncoder,
    _has_cjk,
    _norm_word,
    build_index,
    default_layers,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_build_script():
    """Import scripts/build_retrieval_index.py by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "build_retrieval_index", _REPO_ROOT / "scripts" / "build_retrieval_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

D = 16
LAYERS = [1, 2, 3]

FAKE_CORPUS = [
    ("a red car on the street", "vg"),
    ("a worker wearing a hard hat", "vg"),
    ("a wet floor warning sign", "vg"),
    ("an open flame near fuel", "scenario:general"),
    ("a cracked concrete wall", "scenario:construction"),
    ("a fire extinguisher on the wall", "scenario:general"),
    ("a broken glass bottle", "vg"),
    ("a spilled chemical container", "scenario:pharmaceutical"),
    ("a tall ladder against a building", "scenario:construction"),
    ("a group of people meeting", "vg"),
]


def _fake_encode_factory(seed: int = 0):
    """Deterministic random-embedding stub matching the `encode_fn` seam:
    text -> (metas, hiddens). No model/tokenizer involved."""
    gen = torch.Generator().manual_seed(seed)

    def encode(text: str):
        words = text.split()
        # emulate skipping BOS + pos 1: drop the first two "tokens"
        kept = words[2:] if len(words) > 2 else words
        metas = []
        for i, w in enumerate(kept):
            metas.append({"token_str": " " + w, "word": w,
                          "position": i + 2, "token_id": 1000 + i})
        n = len(kept)
        hiddens = {L: torch.randn(n, D, generator=gen) for L in LAYERS}
        return metas, hiddens

    return encode


def test_default_layers_depth32():
    assert default_layers(32) == [1, 2, 4, 8, 16, 24, 30, 31]
    # works for a shallow model too (no negative / duplicate indices)
    dl = default_layers(4)
    assert dl == sorted(set(dl)) and all(0 <= l < 4 for l in dl)


def _build_tiny():
    return build_index(jl=None, corpus=FAKE_CORPUS, layers=LAYERS,
                       reservoir_cap=50, encode_fn=_fake_encode_factory(),
                       show_progress=False)


def test_build_and_shapes():
    index = _build_tiny()
    assert index.available_layers == LAYERS
    assert index.hidden_dim == D
    assert len(index) > 0
    # every stored vector is unit-norm
    for L in LAYERS:
        V = index._layers_data[L]["vectors"].float()
        norms = V.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
        # metadata rows align with vectors and carry provenance
        meta = index._layers_data[L]["meta"]
        assert len(meta) == V.shape[0]
        assert all({"word", "token_str", "source_sentence", "source_tag",
                    "position", "token_id"} <= set(m) for m in meta)


def test_save_load_roundtrip(tmp_path):
    index = _build_tiny()
    p = tmp_path / "idx.pt"
    index.save(p)
    loaded = RetrievalIndex.load(p)
    assert loaded.available_layers == index.available_layers
    assert len(loaded) == len(index)
    assert loaded.hidden_dim == index.hidden_dim


def test_read_returns_ranked_neighbors():
    index = _build_tiny()
    lens = RetrievalLens(index)
    q = torch.randn(D)
    # all-layers cross-merge
    neigh = lens.read(q, k=5)
    assert 1 <= len(neigh) <= 5
    assert all(isinstance(n, Neighbor) for n in neigh)
    # scores are cosine sims in [-1, 1] and sorted best-first
    scores = [n.score for n in neigh]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)
    assert all(n.source_tag and n.source_sentence for n in neigh)
    # single-layer restriction
    single = lens.read(q, layer=LAYERS[0], k=3)
    assert all(n.layer == LAYERS[0] for n in single)


def test_search_dict_query_per_layer():
    index = _build_tiny()
    q_by = {L: torch.randn(D) for L in LAYERS}
    res = index.search(q_by, k=4)
    assert len(res) == 1 and len(res[0]) <= 4


def test_propose_aggregates_words(monkeypatch):
    index = _build_tiny()
    lens = RetrievalLens(index)

    # Stub _capture so propose needs no model: return a per-layer query vector.
    def fake_capture(jl, prompt, position, layers):
        return {L: torch.randn(D) for L in layers}

    monkeypatch.setattr(RetrievalLens, "_capture", staticmethod(fake_capture))

    words = lens.propose(jl=None, prompt="anything", k=5)
    assert 1 <= len(words) <= 5
    assert all({"word", "score", "source_tag", "count"} <= set(w) for w in words)
    # ranked best-first, words unique
    ws = [w["word"].lower() for w in words]
    assert len(ws) == len(set(ws))
    assert [w["score"] for w in words] == sorted((w["score"] for w in words),
                                                 reverse=True)

    # aggregate=False yields raw neighbors
    raw = lens.propose(jl=None, prompt="anything", k=3, aggregate=False)
    assert all(isinstance(n, Neighbor) for n in raw) and len(raw) <= 3


def test_load_rejects_bad_schema(tmp_path):
    """load() runs under weights_only=True and rejects a wrong/missing schema
    version instead of trusting arbitrary contents."""
    p = tmp_path / "bad.pt"
    torch.save({"version": 999, "dtype": "torch.float32",
                "layers_data": {"1": {"vectors": torch.randn(3, D),
                                      "meta": [{}, {}, {}]}}}, p)
    with pytest.raises(ValueError):
        RetrievalIndex.load(p)


# --- fail-closed behavior of the build script ------------------------------
def test_vg_corpus_fail_closed(tmp_path):
    mod = _load_build_script()
    missing = tmp_path / "nope.txt"
    # missing VG file with no escape hatch -> hard error (not a silent placeholder)
    with pytest.raises(SystemExit):
        mod.load_vg_corpus(str(missing), limit=None)
    # explicit opt-in yields the placeholder list, tagged 'vg'
    placeholder = mod.load_vg_corpus(str(missing), limit=None, allow_placeholder=True)
    assert placeholder and all(tag == "vg" for _, tag in placeholder)
    # a real file is read and tagged 'vg'
    real = tmp_path / "vg.txt"
    real.write_text("a red car\na blue truck\n")
    got = mod.load_vg_corpus(str(real), limit=None)
    assert [t for t, _ in got] == ["a red car", "a blue truck"]


def _crafted_index():
    """A hand-built 1-layer index with known words and orthogonal unit vectors,
    so a chosen query gives a deterministic neighbor ranking."""
    words = ["cigarette", "▁cigarette", "vape", "smoking", "tobacco"]
    V = torch.eye(len(words))  # unit-norm, orthogonal rows
    meta = [{"word": w, "token_str": w, "source_sentence": "a photo of " + w,
             "source_tag": "vg", "position": 2, "token_id": i}
            for i, w in enumerate(words)]
    return RetrievalIndex({1: {"vectors": V, "meta": meta}}), words


def test_exclude_self_filters_echoes(monkeypatch):
    index, words = _crafted_index()
    lens = RetrievalLens(index)
    # query ranks cigarette > ▁cigarette > vape > smoking > tobacco
    q = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    monkeypatch.setattr(RetrievalLens, "_capture",
                        staticmethod(lambda jl, prompt, position, layers: {1: q}))

    # echoes kept -> the concept word itself is top
    kept = lens.propose(jl=None, prompt="cigarette", k=5, exclude_self=False)
    assert kept[0]["word"].lower() == "cigarette"

    # exclude_self drops both "cigarette" AND the marker variant "▁cigarette"
    filt = lens.propose(jl=None, prompt="cigarette", k=5,
                        exclude_self=True, exclude=["cigarette"])
    got = {w["word"].lower().replace("▁", "") for w in filt}
    assert "cigarette" not in got
    assert {"vape", "smoking", "tobacco"} <= got

    # propose_concept excludes the concept by default (surfaces alternatives)
    alt = lens.propose_concept(jl=None, concept="cigarette", k=5)
    assert all(w["word"].lower().replace("▁", "") != "cigarette" for w in alt)


def test_propose_rendered_uses_chat_template(monkeypatch):
    index, _ = _crafted_index()
    lens = RetrievalLens(index)
    q = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

    rendered_seen = {}

    class FakeTok:
        chat_template = "{{ messages }}"  # non-empty => propose_rendered proceeds

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True, **kw):
            rendered_seen["msg"] = messages
            return "<|im_start|>user\n" + messages[-1]["content"] + "<|im_end|>"

    class FakeJL:
        tok = FakeTok()

    def fake_capture(jl, prompt, position, layers):
        rendered_seen["prompt"] = prompt  # confirm it got the rendered string
        return {1: q}

    monkeypatch.setattr(RetrievalLens, "_capture", staticmethod(fake_capture))

    msgs = [{"role": "user", "content": "a person holding a cigarette"}]
    words = lens.propose_rendered(FakeJL(), msgs, position=-1, k=3)
    assert 1 <= len(words) <= 3
    assert all({"word", "score"} <= set(w) for w in words)
    # it rendered via the chat template and fed the rendered string to retrieval
    assert rendered_seen["prompt"].startswith("<|im_start|>")
    assert rendered_seen["msg"] == msgs


def test_propose_rendered_requires_chat_template(monkeypatch):
    index, _ = _crafted_index()
    lens = RetrievalLens(index)

    class NoTemplateTok:
        chat_template = None

    class FakeJL:
        tok = NoTemplateTok()

    with pytest.raises(ValueError, match="chat template"):
        lens.propose_rendered(FakeJL(), [{"role": "user", "content": "hi"}])


def test_norm_word_preserves_identity_symbols():
    # word-internal / trailing identity symbols survive; wrapping punct stripped
    assert _norm_word("C++") == "c++"
    assert _norm_word("C#") == "c#"
    assert _norm_word("F#") == "f#"
    assert _norm_word('"C"') == "c"
    assert _norm_word("(vape)") == "vape"
    assert _norm_word("▁cigarette") == "cigarette"
    assert _norm_word("-cigarette") == "cigarette"


def test_exclude_self_does_not_collapse_cpp(monkeypatch):
    words = ["C++", "C#", "Java"]
    V = torch.eye(len(words))
    meta = [{"word": w, "token_str": w, "source_sentence": "code in " + w,
             "source_tag": "vg", "position": 2, "token_id": i}
            for i, w in enumerate(words)]
    lens = RetrievalLens(RetrievalIndex({1: {"vectors": V, "meta": meta}}))
    q = torch.tensor([3.0, 2.0, 1.0])
    monkeypatch.setattr(RetrievalLens, "_capture",
                        staticmethod(lambda jl, prompt, position, layers: {1: q}))

    # reading a bare "C" token must NOT wipe out "C++"/"C#" as echoes
    got = {w["word"] for w in lens.propose(jl=None, prompt="C", k=3,
                                           exclude=["C"])}
    assert {"C++", "C#", "Java"} <= got
    # but excluding "C++" exactly still removes it
    got2 = {w["word"] for w in lens.propose(jl=None, prompt="C", k=3,
                                            exclude=["C++"])}
    assert "C++" not in got2 and "C#" in got2


def test_iterative_overfetch_backfills_alternatives(monkeypatch):
    # 40 self-echo entries rank above 3 real alternatives; a single k*8 fetch
    # would return nothing after filtering — iterative widening must reach them.
    words = ["cigarette"] * 40 + ["vape", "smoking", "tobacco"]
    n = len(words)
    V = torch.eye(n)
    meta = [{"word": w, "token_str": w, "source_sentence": "s",
             "source_tag": "vg", "position": 2, "token_id": i}
            for i, w in enumerate(words)]
    lens = RetrievalLens(RetrievalIndex({1: {"vectors": V, "meta": meta}}))
    q = torch.tensor([float(n - i) for i in range(n)])  # descending by row
    monkeypatch.setattr(RetrievalLens, "_capture",
                        staticmethod(lambda jl, prompt, position, layers: {1: q}))

    got = lens.propose(jl=None, prompt="cigarette", k=3, exclude=["cigarette"])
    names = {w["word"] for w in got}
    assert len(got) == 3
    assert names == {"vape", "smoking", "tobacco"}


def test_capture_position_out_of_range():
    index, _ = _crafted_index()
    lens = RetrievalLens(index)

    class FakeLM:
        def encode(self, prompt):
            return torch.zeros(1, 3, dtype=torch.long)  # 3-token sequence

    class FakeJL:
        lm = FakeLM()

    with pytest.raises(ValueError, match="out of range"):
        lens._capture(FakeJL(), "x", 5, [1])
    with pytest.raises(ValueError, match="out of range"):
        lens._capture(FakeJL(), "x", -4, [1])


def test_scenario_fetch_fail_closed(monkeypatch):
    mod = _load_build_script()

    def boom(host, remote_cmd, timeout=30):
        raise RuntimeError("ssh down")

    monkeypatch.setattr(mod, "_remote", boom)
    # strict (default): a failed fetch raises rather than dropping domain data
    with pytest.raises(SystemExit):
        mod.load_scenario_corpus(strict=True)
    # explicit opt-out: warn and return empty
    assert mod.load_scenario_corpus(strict=False) == []


# --- CJK-aware subword -> surface-word grouping ----------------------------
class _FakeTok:
    """Minimal ``decode`` over a token-string vocab; ``▁`` -> space like a
    SentencePiece tokenizer, so joined subwords strip back to a clean word."""

    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids).replace("▁", " ")


class _FakeEnc:
    def __init__(self, word_ids):
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids


class _FakeJL:
    def __init__(self, tok):
        self.tok = tok


def _mk_word_map(tokens_with_wid):
    """Build (_ModelEncoder, word_of_pos) from a list of (token_str, word_id),
    with no model/tokenizer download — exercises _word_map on the CPU."""
    vocab = [t for t, _ in tokens_with_wid]
    ids = list(range(len(vocab)))
    word_ids = [w for _, w in tokens_with_wid]
    enc = _FakeEnc(word_ids)
    encoder = _ModelEncoder(_FakeJL(_FakeTok(vocab)), layers=[1])
    return encoder, encoder._word_map(enc, ids)


def test_has_cjk_detection():
    assert _has_cjk("视障乘客")
    assert _has_cjk("走路老伯")
    assert not _has_cjk("fire extinguisher")
    assert not _has_cjk("C++ 123 !?")
    assert _has_cjk("mixed 中 text")  # any CJK char is enough


def test_word_map_cjk_run_not_collapsed_to_one_word():
    # a spaceless Chinese run the pre-tokenizer lumps under a single word_id
    tokens = [("视", 0), ("障", 0), ("乘", 0), ("客", 0), ("手", 0), ("持", 0)]
    _, wm = _mk_word_map(tokens)
    surfaces = set(wm.values())
    # it must NOT surface the whole sentence as one giant "word"
    assert "视障乘客手持" not in surfaces
    # per-token fallback => every surface is short & clean
    assert all(len(s) <= _ModelEncoder.CJK_GROUP_CAP for s in surfaces)
    assert surfaces == {"视", "障", "乘", "客", "手", "持"}


def test_word_map_english_grouping_unchanged():
    # subwords of one English word still merge into the whole surface word
    tokens = [("▁fire", 0), ("▁ext", 1), ("inguisher", 1), ("▁sign", 2)]
    _, wm = _mk_word_map(tokens)
    assert wm[0] == "fire"
    assert wm[1] == "extinguisher" and wm[2] == "extinguisher"
    assert wm[3] == "sign"


def test_word_map_mixed_en_zh_is_sane():
    # english word grouped; a short 2-char CJK word stays whole; a long CJK run
    # splits to per-token instead of collapsing into a fragment
    tokens = [("▁man", 0),
              ("老", 1), ("伯", 1),
              ("视", 2), ("障", 2), ("乘", 2), ("客", 2), ("手", 2), ("持", 2)]
    _, wm = _mk_word_map(tokens)
    assert wm[0] == "man"
    # short CJK word (<= cap) is preserved as a unit
    assert wm[1] == "老伯" and wm[2] == "老伯"
    # long CJK run is broken up, never surfaced whole
    long_surfaces = {wm[p] for p in range(3, 9)}
    assert "视障乘客手持" not in long_surfaces
    assert long_surfaces == {"视", "障", "乘", "客", "手", "持"}


def test_word_map_none_without_word_ids():
    encoder = _ModelEncoder(_FakeJL(_FakeTok(["a"])), layers=[1])
    assert encoder._word_map(_FakeEnc(None), [0]) is None

    class NoWordIds:  # tokenizer exposing no word_ids at all
        def word_ids(self, i=0):
            raise AttributeError("slow tokenizer has no word_ids")

    assert encoder._word_map(NoWordIds(), [0]) is None
