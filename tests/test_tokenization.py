"""JLensVL tokenizer-only helpers: _word_ids / _decode. No model weights needed.

`JLensVL.__init__` only touches `model.config` via getattr with a default, so
passing `model=None` is safe and lets us exercise `_word_ids`/`_decode`
without ever constructing a real (multimodal) model.
"""
from __future__ import annotations

import types

import pytest

from jlensvl import JLensVL

STUB_LM = types.SimpleNamespace(n_layers=36, d_model=2560)


def _make_jlvl(tokenizer):
    return JLensVL(model=None, processor=tokenizer, lm=STUB_LM, lens=None)


class _StubTokenizer:
    """Deterministic fake tokenizer for testing `_decode`'s newline handling
    without depending on the real Qwen vocabulary."""

    def decode(self, ids):
        table = {1: "hello", 2: "\n", 3: "wor\nld"}
        return "".join(table.get(i, "?") for i in ids)


def test_constructs_without_a_model():
    jl = _make_jlvl(_StubTokenizer())
    assert jl.model is None
    assert jl.image_token_id is None
    assert jl.n_layers == STUB_LM.n_layers
    assert jl.d_model == STUB_LM.d_model
    assert jl.tok is not None


def test_decode_replaces_newline_with_literal_backslash_n():
    jl = _make_jlvl(_StubTokenizer())
    out = jl._decode([1, 2, 3])
    assert out == ["hello", "\\n", "wor\\nld"]
    # confirm no *actual* newline survived
    assert all("\n" not in s for s in out)


def test_decode_returns_list_of_strings():
    jl = _make_jlvl(_StubTokenizer())
    out = jl._decode([1, 1, 1])
    assert isinstance(out, list)
    assert all(isinstance(s, str) for s in out)
    assert len(out) == 3


def test_word_ids_nonempty_sorted_unique(qwen_tokenizer):
    jl = _make_jlvl(qwen_tokenizer)
    ids = jl._word_ids(["cat", "dog"])

    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_word_ids_empty_for_no_words(qwen_tokenizer):
    jl = _make_jlvl(qwen_tokenizer)
    assert jl._word_ids([]) == []


def test_decode_of_word_ids_yields_nonempty_strings(qwen_tokenizer):
    jl = _make_jlvl(qwen_tokenizer)
    ids = jl._word_ids(["cat", "dog"])
    decoded = jl._decode(ids)

    assert len(decoded) == len(ids)
    assert all(isinstance(s, str) for s in decoded)
    # each single-token id should decode to something non-empty
    assert all(len(s) > 0 for s in decoded)


def test_word_ids_round_trip_single_token(qwen_tokenizer):
    """Every id returned by _word_ids for 'cat'/'dog' must itself be a
    single-token encoding of one of the requested variants."""
    jl = _make_jlvl(qwen_tokenizer)
    ids = jl._word_ids(["cat"])
    assert len(ids) > 0
    for i in ids:
        decoded = qwen_tokenizer.decode([i])
        assert decoded.strip().lower() == "cat"
