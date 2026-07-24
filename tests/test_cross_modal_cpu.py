"""CPU stub tests for the pure-math pieces of the vision-side tracks.

No model weights / GPU needed: they exercise the rectangular cross-modal Jacobian
estimator and the batched per-patch swap edit against closed forms.

    python tests/test_cross_modal_cpu.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jlensvl.cross_modal import (cross_modal_jacobian_rows, CrossModalJacobianLens,  # noqa
                                 combine_cross_modal)
from jlensvl.interventions import batched_swap_delta, LensIntervention  # noqa


def test_rectangular_estimator():
    """target = M @ (source @ W.T); check row c against the closed form
    Jx[c,:] = W[c,:] * mean_p sum_{p' in pos} M[p',p]."""
    torch.manual_seed(0)
    S, Cs, T, Ct = 5, 3, 4, 6
    W = torch.randn(Ct, Cs, dtype=torch.float64)
    M = torch.randn(T, S, dtype=torch.float64)
    src = torch.randn(S, Cs, dtype=torch.float64, requires_grad=True)
    tgt = M @ (src @ W.T)                                  # [T, Ct]
    pos = torch.tensor([0, 2, 3])                          # subset of target rows
    rows = cross_modal_jacobian_rows(src, tgt, target_positions=pos, retain_last=True)
    coeff = M[pos].sum(dim=0).mean()                       # scalar: mean_p sum_{p'} M
    expected = W * coeff                                   # [Ct, Cs]
    err = (rows.double() - expected).abs().max().item()
    assert err < 1e-5, f"rectangular estimator mismatch: {err}"   # rows forced fp32
    # channel subset returns only those rows, same values (reuses the retained graph).
    ch = torch.tensor([1, 4])
    sub = cross_modal_jacobian_rows(src, tgt, target_positions=pos, channels=ch)
    assert torch.allclose(sub.double(), expected[ch], atol=1e-5)
    print("[ok] rectangular estimator matches closed form (err=%.2e)" % err)


def test_cross_modal_lens_roundtrip(tmp="/tmp/_xlens_test.pt"):
    Jx = torch.randn(6, 3)
    lens = CrossModalJacobianLens(Jx, source_block=7, d_llm=6, d_vision=3)
    lens.save(tmp)
    back = CrossModalJacobianLens.load(tmp)
    p = torch.randn(3)
    assert torch.allclose(lens.transport(p), back.transport(p).float(), atol=1e-2)
    assert torch.allclose(lens.transport(p), p @ Jx.T, atol=1e-4)
    two = combine_cross_modal([lens, lens])                # mean of identical -> same
    assert two.n_samples == 2 and torch.allclose(two.Jx, Jx, atol=1e-4)
    os.remove(tmp)
    print("[ok] CrossModalJacobianLens save/load/transport/combine")


def test_batched_swap_matches_scalar():
    """batched_swap_delta must equal LensIntervention._swap_delta per patch."""
    torch.manual_seed(1)
    S, d = 7, 5
    H = torch.randn(S, d)
    u_a = torch.randn(S, d)
    u_b = torch.randn(S, d)
    t = 0.6
    batched = batched_swap_delta(H, u_a, u_b, t)
    for p in range(S):
        ref = LensIntervention._swap_delta(H[p], u_a[p], u_b[p], t)
        err = (batched[p] - ref).abs().max().item()
        assert err < 1e-4, f"patch {p} swap mismatch: {err}"
    print("[ok] batched_swap_delta matches scalar _swap_delta per patch")


def test_swap_actually_swaps():
    """At t=1 the readings (u_a.h, u_b.h) should exchange (orthonormal dirs)."""
    torch.manual_seed(2)
    S, d = 4, 6
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    u_a = Q[0].repeat(S, 1); u_b = Q[1].repeat(S, 1)       # orthonormal per patch
    H = torch.randn(S, d)
    a0 = (u_a * H).sum(-1); b0 = (u_b * H).sum(-1)
    Hn = H + batched_swap_delta(H, u_a, u_b, 1.0)
    a1 = (u_a * Hn).sum(-1); b1 = (u_b * Hn).sum(-1)
    assert torch.allclose(a1, b0, atol=1e-4) and torch.allclose(b1, a0, atol=1e-4)
    print("[ok] t=1 full swap exchanges the two lens readings")


if __name__ == "__main__":
    test_rectangular_estimator()
    test_cross_modal_lens_roundtrip()
    test_batched_swap_matches_scalar()
    test_swap_actually_swaps()
    print("\nall cross-modal CPU stub tests passed")
