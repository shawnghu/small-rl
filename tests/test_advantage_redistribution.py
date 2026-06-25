"""Unit tests for graft_advantages.compute_advantages / adapter_kappas.

Verifies the GRAFT advantage-redistribution spec (plan §1-§3):
  - baseline from non-detected, scale from full group
  - λ=0 recovers no-routing; λ=1 clean routing (a_X=0, a_Y=κ_Y·â)
  - classic vs exclusive tables
  - all-detected group → plain GRPO fallback
  - active-policy invariant a_R·n_R + a_F·n_F = (n_R+n_F)·â  (equal AND unequal adapters)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graft_advantages import adapter_kappas, compute_advantages  # noqa: E402

G = 4
# group 0: mixed (idx 3 detected); group 1: all-detected
REWARDS = torch.tensor([0., 1., 2., 3., 5., 6., 7., 8.])
IS_RH = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
G0 = slice(0, 4)
G1 = slice(4, 8)


def test_kappas_equal_and_unequal():
    assert adapter_kappas(64, 64) == (2.0, 2.0)
    kr, kf = adapter_kappas(64, 16)
    assert abs(kr - 1.25) < 1e-9 and abs(kf - 5.0) < 1e-9


def test_lambda0_is_no_routing():
    for mode in ("classic", "exclusive"):
        ah, aR, aF = compute_advantages(REWARDS, IS_RH, G, mode, lam=0.0,
                                        kappa_r=2.0, kappa_f=2.0)
        assert torch.allclose(aR, ah) and torch.allclose(aF, ah), mode


def test_classic_clean_routing():
    ah, aR, aF = compute_advantages(REWARDS, IS_RH, G, "classic", lam=1.0,
                                    kappa_r=2.0, kappa_f=2.0)
    # mixed group: detected idx3 -> retain 0, forget 2·â ; non-detected unchanged
    assert torch.allclose(aR[3], torch.zeros(()))
    assert torch.allclose(aF[3], 2 * ah[3])
    assert torch.allclose(aR[0:3], ah[0:3]) and torch.allclose(aF[0:3], ah[0:3])


def test_exclusive_both_directions():
    ah, aR, aF = compute_advantages(REWARDS, IS_RH, G, "exclusive", lam=1.0,
                                    kappa_r=2.0, kappa_f=2.0)
    # mixed group: non-detected -> retain 2·â, forget 0 ; detected -> retain 0, forget 2·â
    assert torch.allclose(aR[0], 2 * ah[0]) and torch.allclose(aF[0], torch.zeros(()))
    assert torch.allclose(aR[3], torch.zeros(())) and torch.allclose(aF[3], 2 * ah[3])


def test_all_detected_group_falls_back_to_plain_grpo():
    for mode in ("classic", "exclusive"):
        ah, aR, aF = compute_advantages(REWARDS, IS_RH, G, mode, lam=1.0,
                                        kappa_r=2.0, kappa_f=2.0)
        # group 1 is all-detected -> no redistribution
        assert torch.allclose(aR[G1], ah[G1]) and torch.allclose(aF[G1], ah[G1]), mode
        # all-detected group is plain-centered: mean advantage 0
        assert abs(float(ah[G1].mean())) < 1e-5


def test_active_policy_invariant_equal_and_unequal():
    cases = [("classic", 64, 64), ("exclusive", 64, 64),
             ("classic", 64, 16), ("exclusive", 16, 64)]
    for mode, nR, nF in cases:
        kr, kf = adapter_kappas(nR, nF)
        ah, aR, aF = compute_advantages(REWARDS, IS_RH, G, mode, lam=1.0,
                                        kappa_r=kr, kappa_f=kf)
        # mixed group (group 0) preserves joint active-policy magnitude per sample
        lhs = aR[G0] * nR + aF[G0] * nF
        rhs = (nR + nF) * ah[G0]
        assert torch.allclose(lhs, rhs, atol=1e-5), (mode, nR, nF)


def test_zero_variance_group_is_finite():
    r = torch.tensor([2., 2., 2., 2.])  # σ=0
    d = torch.tensor([0, 0, 0, 0])
    ah, aR, aF = compute_advantages(r, d, 4, "classic", lam=1.0)
    assert torch.isfinite(ah).all() and torch.allclose(ah, torch.zeros(4))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all advantage-redistribution tests passed")
