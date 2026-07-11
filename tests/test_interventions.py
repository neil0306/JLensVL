"""Unit tests for the lens-coordinate swap math (no model required)."""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
for _p in ("/home/anu/src/ning/JLensVL/src",
           "/home/anu/src/ning/J-space-test/jacobian-lens"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import pytest

from jlensvl.interventions import LensIntervention

swap_delta = LensIntervention._swap_delta


def _rand(d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(d, generator=g), torch.randn(d, generator=g),
            torch.randn(d, generator=g))


def test_full_swap_exchanges_lens_readings():
    for seed in range(5):
        u_a, u_b, h = _rand(seed=seed)
        a0, b0 = float(u_a @ h), float(u_b @ h)
        delta = swap_delta(h, u_a, u_b, 1.0)
        h2 = h + delta
        assert float(u_a @ h2) == pytest.approx(b0, abs=1e-3)
        assert float(u_b @ h2) == pytest.approx(a0, abs=1e-3)


def test_zero_dose_is_noop():
    u_a, u_b, h = _rand(seed=3)
    delta = swap_delta(h, u_a, u_b, 0.0)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_dose_interpolates_linearly():
    u_a, u_b, h = _rand(seed=7)
    a0, b0 = float(u_a @ h), float(u_b @ h)
    for t in (0.25, 0.5, 0.75):
        h2 = h + swap_delta(h, u_a, u_b, t)
        # reading along a moves fraction t of the way from a0 toward b0
        assert float(u_a @ h2) == pytest.approx(a0 + t * (b0 - a0), abs=1e-3)
        assert float(u_b @ h2) == pytest.approx(b0 + t * (a0 - b0), abs=1e-3)


def test_edit_lives_in_span_of_the_two_directions():
    u_a, u_b, h = _rand(seed=11)
    delta = swap_delta(h, u_a, u_b, 1.0)
    U = torch.stack([u_a, u_b], dim=1)               # [d,2]
    # projection of delta onto span(U) should recover delta (delta in the span)
    proj = U @ torch.linalg.lstsq(U, delta[:, None]).solution
    assert torch.allclose(proj.squeeze(1), delta, atol=1e-3)


def test_near_parallel_directions_do_not_blow_up():
    _, _, h = _rand(seed=1)
    u_a = torch.randn(32)
    u_b = u_a + 1e-4 * torch.randn(32)               # almost parallel
    delta = swap_delta(h, u_a, u_b, 1.0)
    assert torch.isfinite(delta).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
