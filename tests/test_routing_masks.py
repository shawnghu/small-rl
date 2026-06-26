"""graft-port first slice: λ/κ redistribution masks + κ-derivation + fail-loud
geometry guard (advantages.py). Pure-function, CPU, no trainer/torch-model.

Pins: (1) master's hardcoded masks == the λ=1/κ=2 special case; (2) the soft-routing
formulas; (3) λ>1 raises; (4) the mode-aware (forget-classic vs retain-exclusive),
λ-aware fail-loud geometry guard; (5) the equal-pressure identity for unequal κ.
"""
import math

import pytest
import torch

from advantages import (GRAFT_W_MAX, LAM_CAP_MARGIN, adapter_kappas,
                        assert_kappa_geometry, kappa_abs, per_group_lam_eff,
                        per_sample_routing_weights, routing_grad_mask_weights)


def test_adapter_kappas():
    assert adapter_kappas(64, 64) == (2.0, 2.0)
    assert adapter_kappas(8, 120) == (128 / 8, 128 / 120)      # tiny retain
    assert adapter_kappas(120, 8) == (128 / 120, 128 / 8)      # tiny forget (localization)
    with pytest.raises(AssertionError):
        adapter_kappas(0, 64)


def test_master_special_case_lambda1_kappa2():
    # master classic: good (rgm=1,fgm=1), bad (rgm=0, fgm=forget_bad_scale=2)
    rg, fg, rb, fb = routing_grad_mask_weights("classic", 1.0, 2.0, 2.0)
    assert (rg, fg, rb, fb) == (1.0, 1.0, 0.0, 2.0)
    # master exclusive bad is also (0,2); good fgm=0; good rgm is master's stub 1 but
    # the port COMPENSATES it to κ_R=2 (the fix).
    rg, fg, rb, fb = routing_grad_mask_weights("exclusive", 1.0, 2.0, 2.0)
    assert (rg, fg, rb, fb) == (2.0, 0.0, 0.0, 2.0)


def test_soft_routing_formulas():
    # classic λ=0.5, κ_f=3: bad rgm=1-0.5=0.5, fgm=1+0.5*2=2.0; good untouched.
    assert routing_grad_mask_weights("classic", 0.5, 2.0, 3.0) == (1.0, 1.0, 0.5, 2.0)
    # exclusive λ=0.5, κ_r=4, κ_f=3: good (1+0.5*3, 0.5)=(2.5,0.5); bad (0.5, 1+0.5*2)=(0.5,2.0)
    assert routing_grad_mask_weights("exclusive", 0.5, 4.0, 3.0) == (2.5, 0.5, 0.5, 2.0)
    # λ=0 → identity (both adapters weight 1 everywhere)
    assert routing_grad_mask_weights("classic", 0.0, 9.0, 9.0) == (1.0, 1.0, 1.0, 1.0)
    assert routing_grad_mask_weights("exclusive", 0.0, 9.0, 9.0) == (1.0, 1.0, 1.0, 1.0)


def test_lambda_gt1_over_routing_formula():
    # λ>1 is now SUPPORTED (slice 2b): the bad retain mask 1−λ goes negative
    # (anti-train retain on detected) and the bad forget mask amplifies past κ_F.
    # routing_grad_mask_weights is a pure formula (no raise); the per-group cap
    # bounds it. classic λ=1.5, κ_f=2: bad=(1-1.5, 1+1.5*1)=(-0.5, 2.5).
    rg, fg, rb, fb = routing_grad_mask_weights("classic", 1.5, 2.0, 2.0)
    assert (rg, fg, rb, fb) == (1.0, 1.0, -0.5, 2.5)
    # exclusive λ=1.5: good=(1+1.5*1, -0.5)=(2.5,-0.5); bad=(-0.5, 2.5)
    assert routing_grad_mask_weights("exclusive", 1.5, 2.0, 2.0) == (2.5, -0.5, -0.5, 2.5)
    # equal-pressure still holds for equal adapters at λ>1
    assert math.isclose(rb * 1 + fb * 1, 2.0)


def test_per_group_lam_eff_soft_uncapped():
    # λ≤1: every group gets λ (soft routing uncapped), regardless of detection.
    is_rh = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.bool)  # 2 groups, G=4
    le = per_group_lam_eff(is_rh, 4, "classic", 0.5, 2.0, 2.0)
    assert torch.allclose(le, torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_per_group_lam_eff_lower_cap_classic():
    # classic over-routing lower cap: slope=n_det, lower=max(1, 0.95·G/n_det).
    # A group with few detected hits the cap hard. G=4.
    # group0: 1 detected -> slope=1 -> lower=0.95*4/1=3.8 -> min(λ=5, 3.8, upper)
    #         upper=(W_MAX-1)/(κ_F-1)=3/1=3 -> lam_eff=min(5,3.8,3)=3
    # group1: 4 detected -> slope=4 -> lower=0.95 -> max(1,0.95)=1 -> lam_eff=1
    is_rh = torch.tensor([1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)
    le = per_group_lam_eff(is_rh, 4, "classic", 5.0, 2.0, 2.0, w_max=4.0)
    assert math.isclose(le[0].item(), 3.0, rel_tol=1e-9)   # upper-capped
    assert math.isclose(le[1].item(), 1.0, rel_tol=1e-9)   # lower floor=1


def test_per_group_lam_eff_upper_cap_only():
    # With a generous W_MAX the upper cap relaxes; the lower cap then binds.
    is_rh = torch.tensor([1, 0, 0, 0], dtype=torch.bool)   # 1 group, 1 detected
    # slope=1 -> lower=0.95*4=3.8 ; upper=(16-1)/(2-1)=15 ; λ=10 -> min(10,3.8,15)=3.8
    le = per_group_lam_eff(is_rh, 4, "classic", 10.0, 2.0, 2.0, w_max=16.0)
    assert math.isclose(le[0].item(), LAM_CAP_MARGIN * 4 / 1, rel_tol=1e-9)


def test_per_sample_routing_weights_lambda1_matches_scalar():
    # At λ=1 the per-sample weights must equal the scalar masks broadcast by class.
    is_rh = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
    rw, fw, le = per_sample_routing_weights(is_rh, 4, "classic", 1.0, 2.0, 2.0)
    rg, fg, rb, fb = routing_grad_mask_weights("classic", 1.0, 2.0, 2.0)
    assert torch.allclose(rw, torch.tensor([rg, rg, rb, rb]))
    assert torch.allclose(fw, torch.tensor([fg, fg, fb, fb]))


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        routing_grad_mask_weights("none", 1.0, 2.0, 2.0)


def test_kappa_abs_mode_aware():
    # classic: only forget absorbs → κ_F (retain never amplifies, even if κ_R huge)
    assert kappa_abs("classic", 16.0, 1.07) == 1.07
    # exclusive: max(κ_R, κ_F)
    assert kappa_abs("exclusive", 16.0, 1.07) == 16.0
    assert kappa_abs("exclusive", 1.07, 16.0) == 16.0


def test_geometry_guard_mode_and_lambda_aware():
    # equal adapters κ=2: passes at λ=1 (w_floor=2 ≤ 4)
    assert assert_kappa_geometry("classic", 1.0, 2.0, 2.0) == 2.0
    assert assert_kappa_geometry("exclusive", 1.0, 2.0, 2.0) == 2.0
    # κ at the W_MAX boundary (=4) passes at λ=1
    assert assert_kappa_geometry("classic", 1.0, 2.0, 4.0) == 4.0

    # SMALL FORGET in CLASSIC (the user's point): κ_F=16 → fails at λ=1
    with pytest.raises(AssertionError):
        assert_kappa_geometry("classic", 1.0, 1.07, 16.0)
    # SMALL RETAIN in CLASSIC: κ_R=16 but retain never amplifies in classic → PASSES
    assert assert_kappa_geometry("classic", 1.0, 16.0, 1.07) == 1.07
    # SMALL RETAIN in EXCLUSIVE: κ_R=16 → fails (exclusive amplifies retain on good)
    with pytest.raises(AssertionError):
        assert_kappa_geometry("exclusive", 1.0, 16.0, 1.07)

    # λ-aware: κ_abs=16 passes at λ=0.2 (w_floor=1+0.2·15=4 ≤ 4) but fails at λ=0.25
    assert assert_kappa_geometry("classic", 0.2, 1.07, 16.0) == 16.0
    with pytest.raises(AssertionError):
        assert_kappa_geometry("classic", 0.25, 1.07, 16.0)

    # explicit larger W_MAX lets a strongly-unequal adapter through (opt-in)
    assert assert_kappa_geometry("classic", 1.0, 1.07, 16.0, w_max=16.0) == 16.0


def test_equal_pressure_identity_unequal_kappa():
    # w_R·n_R + w_F·n_F == n_R + n_F per sample, any λ≤1, both modes, unequal sizes.
    for n_R, n_F in [(64, 64), (8, 120), (120, 8), (32, 16)]:
        kr, kf = adapter_kappas(n_R, n_F)
        total = n_R + n_F
        for lam in (0.0, 0.5, 1.0):
            for mode in ("classic", "exclusive"):
                rg, fg, rb, fb = routing_grad_mask_weights(mode, lam, kr, kf)
                # good sample (non-detected) and bad sample (detected)
                assert math.isclose(rg * n_R + fg * n_F, total, rel_tol=1e-9), (mode, lam, "good")
                assert math.isclose(rb * n_R + fb * n_F, total, rel_tol=1e-9), (mode, lam, "bad")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
