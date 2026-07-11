"""Fitting/round-trip/merge math for jlens.JacobianLens (no model required)."""
from __future__ import annotations

import pytest
import torch

from jlens import JacobianLens

# fp16 has ~10 mantissa bits -> relative error per element on the order of
# 2**-11. These tolerances give ample margin above that for O(1)-scale randn.
FP16_ATOL = 1e-2
FP16_RTOL = 1e-2


def test_transport_matches_manual_matmul(random_lens):
    layer = random_lens.source_layers[0]
    J = random_lens.jacobians[layer]
    residual = torch.randn(5, random_lens.d_model)

    expected = residual @ J.T
    actual = random_lens.transport(residual, layer)

    assert torch.equal(actual, expected)


def test_transport_handles_leading_batch_dims(random_lens):
    layer = random_lens.source_layers[-1]
    J = random_lens.jacobians[layer]
    residual = torch.randn(2, 3, random_lens.d_model)

    expected = residual @ J.T
    actual = random_lens.transport(residual, layer)

    assert actual.shape == (2, 3, random_lens.d_model)
    assert torch.equal(actual, expected)


def test_save_load_roundtrip_preserves_metadata(random_lens, tmp_path):
    path = tmp_path / "lens.pt"
    random_lens.save(path)
    reloaded = JacobianLens.load(str(path))

    assert reloaded.source_layers == random_lens.source_layers
    assert reloaded.n_prompts == random_lens.n_prompts
    assert reloaded.d_model == random_lens.d_model


def test_save_load_roundtrip_preserves_jacobians_fp16_tolerance(random_lens, tmp_path):
    path = tmp_path / "lens.pt"
    random_lens.save(path)
    reloaded = JacobianLens.load(str(path))

    for layer in random_lens.source_layers:
        torch.testing.assert_close(
            reloaded.jacobians[layer],
            random_lens.jacobians[layer],
            atol=FP16_ATOL,
            rtol=FP16_RTOL,
        )


def test_load_bogus_file_raises_value_error(tmp_path):
    path = tmp_path / "not_a_lens.pt"
    torch.save({"totally": "unrelated", "n_prompts": 3}, path)

    with pytest.raises(ValueError):
        JacobianLens.load(str(path))


def test_from_pretrained_local_file_matches_load(random_lens, tmp_path):
    path = tmp_path / "lens.pt"
    random_lens.save(path)

    via_from_pretrained = JacobianLens.from_pretrained(str(path))
    via_load = JacobianLens.load(str(path))

    assert via_from_pretrained.source_layers == via_load.source_layers
    assert via_from_pretrained.n_prompts == via_load.n_prompts
    assert via_from_pretrained.d_model == via_load.d_model
    for layer in via_load.source_layers:
        assert torch.equal(via_from_pretrained.jacobians[layer], via_load.jacobians[layer])


def test_merge_is_n_prompts_weighted_mean():
    torch.manual_seed(1)
    d_model = 8
    layer = 0
    J_a = torch.randn(d_model, d_model)
    J_b = torch.randn(d_model, d_model)
    lens_a = JacobianLens({layer: J_a}, n_prompts=1, d_model=d_model)
    lens_b = JacobianLens({layer: J_b}, n_prompts=3, d_model=d_model)

    merged = JacobianLens.merge([lens_a, lens_b])

    assert merged.n_prompts == 4
    expected = 0.25 * J_a + 0.75 * J_b
    torch.testing.assert_close(merged.jacobians[layer], expected, atol=1e-6, rtol=1e-6)


def test_merge_raises_on_mismatched_source_layers():
    d_model = 8
    lens_a = JacobianLens(
        {0: torch.randn(d_model, d_model), 1: torch.randn(d_model, d_model)},
        n_prompts=1, d_model=d_model,
    )
    lens_b = JacobianLens(
        {0: torch.randn(d_model, d_model), 2: torch.randn(d_model, d_model)},
        n_prompts=1, d_model=d_model,
    )

    with pytest.raises(ValueError):
        JacobianLens.merge([lens_a, lens_b])


def test_merge_raises_on_mismatched_d_model():
    lens_a = JacobianLens({0: torch.randn(8, 8)}, n_prompts=1, d_model=8)
    lens_b = JacobianLens({0: torch.randn(4, 4)}, n_prompts=1, d_model=4)

    with pytest.raises(ValueError):
        JacobianLens.merge([lens_a, lens_b])


def test_merge_raises_on_empty_list():
    with pytest.raises(ValueError):
        JacobianLens.merge([])
