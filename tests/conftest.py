"""Shared fixtures for the JLensVL test suite.

Inserts `src/` (JLensVL) and the sibling `jacobian-lens` engine package onto
sys.path so `pytest` works standalone even if PYTHONPATH is not set manually
(the documented run command still works too — this is just a belt-and-braces
fallback for local/CI runs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# --- sys.path setup (before any jlens/jlensvl import) ----------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent  # .../JLensVL
_SRC = _REPO_ROOT / "src"
_ENGINE = _REPO_ROOT.parent / "J-space-test" / "jacobian-lens"

for _p in (_SRC, _ENGINE):
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# --- offline HF env, best-effort (won't override an already-set value) -----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"


def have_weights() -> bool:
    """True only when the caller has explicitly told us model weights are
    available (e.g. once the in-progress download finishes). We deliberately
    do NOT try to load the full model here — that would be slow/heavy and
    could touch the GPU — callers that want a load-and-skip pattern should
    wrap their own attempt in a try/except and call pytest.skip()."""
    return os.environ.get("JLENSVL_HAVE_WEIGHTS") == "1"


@pytest.fixture(scope="session")
def weights_available() -> bool:
    return have_weights()


@pytest.fixture
def random_lens():
    """A small, fast JacobianLens over random matrices for pure lens-math tests."""
    from jlens import JacobianLens

    torch.manual_seed(0)
    d_model = 16
    layers = [4, 8, 12]
    jacobians = {L: torch.randn(d_model, d_model) for L in layers}
    return JacobianLens(jacobians, n_prompts=7, d_model=d_model)


@pytest.fixture(scope="session")
def qwen_tokenizer():
    """The real Qwen3.5-4B tokenizer, loaded offline. Skips (not fails) the
    whole dependent test module if the tokenizer files aren't cached."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - any load failure -> skip
        pytest.skip(f"Qwen3.5-4B tokenizer not loadable offline: {exc}")
    return tok
