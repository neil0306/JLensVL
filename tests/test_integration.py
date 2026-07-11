"""Light integration test against the real Qwen3.5-4B weights + GPU.

SKIPPED by default: the model weights are not yet fully downloaded, and we
must not touch GPU 0. Set `JLENSVL_HAVE_WEIGHTS=1` (once weights land) and
`CUDA_VISIBLE_DEVICES=1` to enable. See conftest.have_weights().
"""
from __future__ import annotations

import os

import pytest

from conftest import QWEN_MODEL_ID, have_weights

pytestmark = pytest.mark.skipif(
    not have_weights(),
    reason=(
        "Qwen3.5-4B weights not available locally yet (download in progress); "
        "set JLENSVL_HAVE_WEIGHTS=1 once they land to enable this test."
    ),
)


def test_from_pretrained_and_trace_minimal():
    """Minimal end-to-end smoke test: load the real model, fit a tiny
    single-layer lens on it (so dimensions always match whatever d_model the
    real model reports), and confirm `.trace()` returns the expected shape."""
    import torch

    from jlens import JacobianLens
    from jlensvl import JLensVL

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", (
        "must run with CUDA_VISIBLE_DEVICES=1 (GPU 0 is off-limits)"
    )

    jl = JLensVL.from_pretrained(QWEN_MODEL_ID, lens=None, device="cuda")

    layer = jl.lm.n_layers // 2
    torch.manual_seed(0)
    tiny_lens = JacobianLens(
        {layer: torch.randn(jl.d_model, jl.d_model)}, n_prompts=1, d_model=jl.d_model
    )
    jl.lens = tiny_lens

    result = jl.trace("The capital of France is", layers=[layer], k=3)

    assert isinstance(result, dict)
    assert set(result.keys()) == {layer}
    assert isinstance(result[layer], list)
    assert len(result[layer]) == 3
    assert all(isinstance(t, str) for t in result[layer])
