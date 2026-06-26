"""graft-port first slice: λ/κ redistribution masks + κ-derivation + fail-loud
geometry guard (advantages.py). Pure-function, CPU, no trainer/torch-model.

Pins: (1) master's hardcoded masks == the λ=1/κ=2 special case; (2) the soft-routing
formulas; (3) λ>1 raises; (4) the mode-aware (forget-classic vs retain-exclusive),
λ-aware fail-loud geometry guard; (5) the equal-pressure identity for unequal κ.
"""
import math

import pytest

from advantages import (GRAFT_W_MAX, adapter_kappas, assert_kappa_geometry,
                        kappa_abs, routing_grad_mask_weights)


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


def test_lambda_gt1_raises():
    for mode in ("classic", "exclusive"):
        with pytest.raises(NotImplementedError):
            routing_grad_mask_weights(mode, 1.5, 2.0, 2.0)


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
