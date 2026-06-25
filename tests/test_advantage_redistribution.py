"""Unit tests for graft_advantages.compute_advantages / adapter_kappas.

Verifies the GRAFT advantage-redistribution spec:
  - retain-weighted baseline => retain update is zero-mean at EVERY lambda
  - lambda=0 -> full-group baseline (vanilla GRPO); lambda=1 -> non-detected baseline (old behaviour)
  - classic vs exclusive routing tables
  - all-detected group -> plain GRPO fallback
  - active-policy (equal-pressure) invariant a_R*n_R + a_F*n_F = (n_R+n_F)*a_hat, all lambda + unequal adapters
  - per-group lambda cap prevents the lambda>1 singularity (Sigma w_R stays >= margin*G), advantages finite
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graft_advantages import adapter_kappas, compute_advantages, LAM_CAP_MARGIN  # noqa: E402

G = 4
# group 0: mixed (idx 3 detected); group 1: all-detected
REWARDS = torch.tensor([0., 1., 2., 3., 5., 6., 7., 8.])
IS_RH = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
G0 = slice(0, 4)
G1 = slice(4, 8)

# A single mixed group with several high-reward hacks (n_det=3, n_nd=5), for the
# zero-mean / cap tests. classic singularity at lambda_sing = G/n_det = 8/3 ≈ 2.667.
MIX_G = 8
MIX_R = torch.tensor([0., 1., 2., 3., 4., 10., 11., 12.])
MIX_D = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])


def test_kappas_equal_and_unequal():
    assert adapter_kappas(64, 64) == (2.0, 2.0)
    kr, kf = adapter_kappas(64, 16)
    assert abs(kr - 1.25) < 1e-9 and abs(kf - 5.0) < 1e-9


def test_lambda0_is_no_routing():
    for mode in ("classic", "exclusive"):
        ah, aR, aF, _ = compute_advantages(REWARDS, IS_RH, G, mode, lam=0.0,
                                           kappa_r=2.0, kappa_f=2.0)
        assert torch.allclose(aR, ah) and torch.allclose(aF, ah), mode


def test_classic_clean_routing():
    ah, aR, aF, _ = compute_advantages(REWARDS, IS_RH, G, "classic", lam=1.0,
                                       kappa_r=2.0, kappa_f=2.0)
    # mixed group: detected idx3 -> retain 0, forget 2·â ; non-detected unchanged
    assert torch.allclose(aR[3], torch.zeros(()))
    assert torch.allclose(aF[3], 2 * ah[3])
    assert torch.allclose(aR[0:3], ah[0:3]) and torch.allclose(aF[0:3], ah[0:3])


def test_exclusive_both_directions():
    ah, aR, aF, _ = compute_advantages(REWARDS, IS_RH, G, "exclusive", lam=1.0,
                                       kappa_r=2.0, kappa_f=2.0)
    # mixed group: non-detected -> retain 2·â, forget 0 ; detected -> retain 0, forget 2·â
    assert torch.allclose(aR[0], 2 * ah[0]) and torch.allclose(aF[0], torch.zeros(()))
    assert torch.allclose(aR[3], torch.zeros(())) and torch.allclose(aF[3], 2 * ah[3])


def test_all_detected_group_falls_back_to_plain_grpo():
    for mode in ("classic", "exclusive"):
        ah, aR, aF, _ = compute_advantages(REWARDS, IS_RH, G, mode, lam=1.0,
                                           kappa_r=2.0, kappa_f=2.0)
        # group 1 is all-detected -> no redistribution
        assert torch.allclose(aR[G1], ah[G1]) and torch.allclose(aF[G1], ah[G1]), mode
        # all-detected group is plain-centered: mean advantage 0
        assert abs(float(ah[G1].mean())) < 1e-5


def test_active_policy_invariant_all_lambda_equal_and_unequal():
    cases = [("classic", 64, 64), ("exclusive", 64, 64),
             ("classic", 64, 16), ("exclusive", 16, 64)]
    for mode, nR, nF in cases:
        kr, kf = adapter_kappas(nR, nF)
        for lam in (0.5, 1.0, 1.5):
            ah, aR, aF, _ = compute_advantages(MIX_R, MIX_D, MIX_G, mode, lam=lam,
                                               kappa_r=kr, kappa_f=kf)
            lhs = aR * nR + aF * nF
            rhs = (nR + nF) * ah
            assert torch.allclose(lhs, rhs, atol=1e-4), (mode, lam, nR, nF)


def test_retain_zero_mean_across_lambda():
    # The retain-weighted baseline makes Sigma a_R == 0 at every lambda (single mixed group).
    for mode in ("classic", "exclusive"):
        for lam in (0.0, 0.3, 0.7, 1.0, 1.5, 2.0):
            ah, aR, aF, _ = compute_advantages(MIX_R, MIX_D, MIX_G, mode, lam=lam)
            assert abs(float(aR.sum())) < 1e-4, (mode, lam, float(aR.sum()))


def test_lambda0_recovers_full_group_baseline():
    # lam=0: b = full-group mean => a_hat is full-group-centered (zero-mean over whole group).
    ah, aR, aF, _ = compute_advantages(MIX_R, MIX_D, MIX_G, "classic", lam=0.0)
    assert abs(float(ah.sum())) < 1e-4


def test_lambda1_uses_nondetected_baseline():
    # lam=1 classic: detected zeroed in retain; non-detected a_hat centered on b_nd, sigma=full std.
    ah, aR, aF, _ = compute_advantages(MIX_R, MIX_D, MIX_G, "classic", lam=1.0)
    nd = MIX_D == 0
    assert torch.allclose(aR[~nd], torch.zeros(int((~nd).sum())))   # detected -> 0 in retain
    sigma = MIX_R.std(correction=0)
    b_nd = MIX_R[nd].mean()
    assert torch.allclose(ah[nd], (MIX_R[nd] - b_nd) / (sigma + 1e-4), atol=1e-5)


def test_cap_prevents_singularity():
    # classic, lam well past the per-group singularity (lam_sing = 8/3 ≈ 2.667): the cap keeps
    # Sigma w_R >= margin*G, so advantages stay finite and retain stays zero-mean.
    for lam in (3.0, 10.0):
        ah, aR, aF, diag = compute_advantages(MIX_R, MIX_D, MIX_G, "classic", lam=lam)
        assert torch.isfinite(aR).all() and torch.isfinite(aF).all(), lam
        assert abs(float(aR.sum())) < 1e-4, (lam, float(aR.sum()))        # still zero-mean
        assert diag["frac_groups_capped"] == 1.0, lam                     # group was capped
        # Sigma w_R / G floored at the margin (≈ exactly margin once lam exceeds the cap).
        assert abs(diag["min_retain_weight_frac"] - LAM_CAP_MARGIN) < 1e-3, diag


def test_no_cap_within_unit_interval():
    # lambda in [0,1] is singularity-free => cap never engages, lam_eff == lam.
    for lam in (0.0, 0.5, 1.0):
        _, _, _, diag = compute_advantages(MIX_R, MIX_D, MIX_G, "classic", lam=lam)
        assert diag["frac_groups_capped"] == 0.0, lam
        assert abs(diag["mean_lam_eff"] - lam) < 1e-6, lam


def test_diag_keys_present_and_float():
    _, _, _, diag = compute_advantages(MIX_R, MIX_D, MIX_G, "classic", lam=1.0)
    for k in ("frac_groups_capped", "min_retain_weight_frac",
              "min_lam_singularity", "mean_lam_eff"):
        assert k in diag and isinstance(diag[k], float), k


def test_zero_variance_group_is_finite():
    r = torch.tensor([2., 2., 2., 2.])  # σ=0
    d = torch.tensor([0, 0, 0, 0])
    ah, aR, aF, _ = compute_advantages(r, d, 4, "classic", lam=1.0)
    assert torch.isfinite(ah).all() and torch.allclose(ah, torch.zeros(4))


def test_retain_zero_mean_large_reward_small_sigma():
    # float32 corner (adversarial review): large reward offset + tiny within-group spread +
    # lam past the cap amplifies the weighted-baseline cancellation. float64 keeps sum(a_R)~0.
    cases = [
        (torch.tensor([9., 9., 9., 9.]), torch.tensor([0, 0, 1, 1]), 4, 3.0),
        (torch.full((8,), 1000.), torch.tensor([0, 1, 1, 1, 1, 1, 1, 1]), 8, 10.0),
        (torch.full((8,), 10000.), torch.tensor([0, 1, 1, 1, 1, 1, 1, 1]), 8, 10.0),
    ]
    for r, d, g, lam in cases:
        ah, aR, aF, _ = compute_advantages(r, d, g, "classic", lam=lam)
        assert torch.isfinite(aR).all() and torch.isfinite(aF).all()
        assert abs(float(aR.sum())) < 1e-3, (float(r[0]), lam, float(aR.sum()))


def test_outputs_on_input_device():
    # the per-group lambda tensors must be created on r.device, else torch.where crashes on
    # GPU (CPU-only tests miss it). lam>1 exercises the cap branch (torch.full + torch.minimum).
    r = torch.tensor([0., 1., 2., 3.])
    d = torch.tensor([0, 0, 0, 1])
    ah, aR, aF, _ = compute_advantages(r, d, 4, "classic", lam=2.0)
    assert ah.device == r.device == aR.device == aF.device


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all advantage-redistribution tests passed")
