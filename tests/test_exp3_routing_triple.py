"""Characterization test for Exp 3 (negative-deployment reinterpretation) of the
per-sample fused-GR triple (forget_fwd_scale, retain_grad_mask, forget_grad_mask).

Exp 3 reinterprets the DEPLOYMENT / coherence config as (retain=1, forget=-1) and
runs the routing passes at (1, n), n in {1, 2}. The triple decision lives in the
pure helper train.fused_routing_triple; this test pins its outputs:

  coherence, coh_forget_grad off -> (coh_forget_scale, 1, 0)   e.g. (-1, 1, 0)
  coherence, coh_forget_grad on  -> (coh_forget_scale, 1, 1)   e.g. (-1, 1, 1)
  routing good (classic, lam=1, kappa=2) -> (n, 1, 1)
  routing bad  (classic, lam=1, kappa=2) -> (n, 0, 2)

n is a FORWARD scale only, so the routing grad masks (rgm/fgm from
routing_grad_mask_weights) are UNCHANGED by n.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import fused_routing_triple  # noqa: E402
from advantages import routing_grad_mask_weights  # noqa: E402


# classic, lambda=1, kappa_r=kappa_f=2 (the m16 / Exp-3 base): good (1,1), bad (0,2).
RGM_GOOD, FGM_GOOD, RGM_BAD, FGM_BAD = routing_grad_mask_weights("classic", 1.0, 2.0, 2.0)


def _triple(kind, *, coh_forget_scale, coh_forget_grad_mask, n, **kw):
    return fused_routing_triple(
        kind,
        coh_forget_scale=coh_forget_scale,
        coh_forget_grad_mask=coh_forget_grad_mask,
        routing_forget_scale=n,
        rgm_good=RGM_GOOD, fgm_good=FGM_GOOD,
        rgm_bad=RGM_BAD, fgm_bad=FGM_BAD,
        **kw,
    )


def test_classic_mask_base_is_master_parity():
    # Sanity: the κ=2/λ=1 classic masks are master's hardcoded {good (1,1), bad (0,2)}.
    assert (RGM_GOOD, FGM_GOOD, RGM_BAD, FGM_BAD) == (1.0, 1.0, 0.0, 2.0)


@pytest.mark.parametrize("n", [1.0, 2.0])
def test_coherence_forget_grad_off(n):
    # Exp 3 OFF: deployment forward (1,-1), forget grad masked off.
    assert _triple("coherence", coh_forget_scale=-1.0, coh_forget_grad_mask=0.0, n=n) \
        == (-1.0, 1.0, 0.0)


@pytest.mark.parametrize("n", [1.0, 2.0])
def test_coherence_forget_grad_on(n):
    # Exp 3 ON: forget adapter ALSO updated on the (1,-1) coherence pass.
    assert _triple("coherence", coh_forget_scale=-1.0, coh_forget_grad_mask=1.0, n=n) \
        == (-1.0, 1.0, 1.0)


@pytest.mark.parametrize("n", [1.0, 2.0])
def test_routing_good_uses_n_forward_scale_unchanged_masks(n):
    # Routing good: forward forget scale = n; grad masks unchanged (1,1).
    assert _triple("good", coh_forget_scale=-1.0, coh_forget_grad_mask=0.0, n=n) \
        == (n, 1.0, 1.0)


@pytest.mark.parametrize("n", [1.0, 2.0])
def test_routing_bad_uses_n_forward_scale_unchanged_masks(n):
    # Routing bad: forward forget scale = n; grad masks unchanged (0,2).
    assert _triple("bad", coh_forget_scale=-1.0, coh_forget_grad_mask=0.0, n=n) \
        == (n, 0.0, 2.0)


def test_default_coherence_is_master_parity():
    # Defaults (coh_forget_scale=0, off, n=1) reproduce the historical (0,1,0)
    # coherence triple and (1, rgm/fgm) routing triples — no behavior change.
    assert _triple("coherence", coh_forget_scale=0.0, coh_forget_grad_mask=0.0, n=1.0) \
        == (0.0, 1.0, 0.0)
    assert _triple("good", coh_forget_scale=0.0, coh_forget_grad_mask=0.0, n=1.0) \
        == (1.0, 1.0, 1.0)
    assert _triple("bad", coh_forget_scale=0.0, coh_forget_grad_mask=0.0, n=1.0) \
        == (1.0, 0.0, 2.0)


@pytest.mark.parametrize("n", [1.0, 2.0])
def test_slow_path_uses_per_sample_weights_at_n(n):
    # On the λ≠1 slow path the routing masks are the per-sample λ/κ redistribution
    # weights; the forward forget scale is still n.
    assert _triple("good", coh_forget_scale=-1.0, coh_forget_grad_mask=0.0, n=n,
                   slow=True, retain_w=0.7, forget_w=1.3) == (n, 0.7, 1.3)


def test_unexpected_kind_raises():
    with pytest.raises(AssertionError):
        _triple("bogus", coh_forget_scale=0.0, coh_forget_grad_mask=0.0, n=1.0)
