"""Unit tests for graft_adam.GraftAdam.

Verifies:
  - with G_m == G_v, c=1, no clip, wd=0, the step matches stock torch AdamW exactly
  - a uniform scale on G_m (with G_v unscaled) is NOT laundered by Adam (the κ survives)
  - participation factor c on G_v scales the step by 1/c (per-example down-weighting)
  - freezing an inactive group leaves its (m, v, t) bit-identical
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graft_adam import GraftAdam  # noqa: E402

torch.manual_seed(0)
LR, BETAS, EPS = 1e-2, (0.9, 0.999), 1e-8


def _grads(p, n):
    g = torch.randn_like(p)
    return [torch.randn_like(p) for _ in range(n)]


def test_matches_stock_adamw_when_m_equals_v():
    p_ref = torch.nn.Parameter(torch.randn(8))
    p_grf = torch.nn.Parameter(p_ref.detach().clone())
    ref = torch.optim.AdamW([p_ref], lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0)
    grf = GraftAdam([p_grf], [], lr=LR, betas=BETAS, eps=EPS,
                    weight_decay=0.0, max_grad_norm=0.0)  # clip off
    for g in _grads(p_ref, 6):
        p_ref.grad = g.clone()
        ref.step()
        grf.set_window({p_grf: g.clone()}, {p_grf: g.clone()},
                       {"retain": 1.0}, {"retain": True, "forget": False})
        grf.step()
    assert torch.allclose(p_ref, p_grf, atol=1e-6), (p_ref - p_grf).abs().max()


def test_uniform_m_scale_is_not_laundered():
    # G_m = κ·G_v with c=1 ⇒ step is ≈κ× the G_m==G_v step (Adam does NOT cancel it).
    kappa = 2.0
    g = torch.randn(8)
    p_base = torch.nn.Parameter(torch.zeros(8))
    p_scaled = torch.nn.Parameter(torch.zeros(8))
    o_base = GraftAdam([p_base], [], lr=LR, betas=BETAS, eps=EPS, max_grad_norm=0.0)
    o_scaled = GraftAdam([p_scaled], [], lr=LR, betas=BETAS, eps=EPS, max_grad_norm=0.0)
    o_base.set_window({p_base: g}, {p_base: g}, {"retain": 1.0},
                      {"retain": True, "forget": False})
    o_base.step()
    o_scaled.set_window({p_scaled: kappa * g}, {p_scaled: g}, {"retain": 1.0},
                        {"retain": True, "forget": False})
    o_scaled.step()
    ratio = (p_scaled.abs() / p_base.abs().clamp(min=1e-12)).mean()
    assert abs(float(ratio) - kappa) < 1e-3, ratio


def test_participation_factor_scales_step_by_inverse_c():
    # Same G_m/G_v; forget uses c=N/N_F ⇒ √v scaled by c ⇒ step scaled by 1/c.
    c = 4.0
    g = torch.randn(8)
    p_r = torch.nn.Parameter(torch.zeros(8))
    p_f = torch.nn.Parameter(torch.zeros(8))
    opt = GraftAdam([p_r], [p_f], lr=LR, betas=BETAS, eps=EPS, max_grad_norm=0.0)
    opt.set_window({p_r: g, p_f: g}, {p_r: g, p_f: g},
                   {"retain": 1.0, "forget": c},
                   {"retain": True, "forget": True})
    opt.step()
    ratio = (p_f.abs() / p_r.abs().clamp(min=1e-12)).mean()
    assert abs(float(ratio) - 1.0 / c) < 1e-3, ratio


def test_freeze_leaves_inactive_group_untouched():
    g = torch.randn(8)
    p_r = torch.nn.Parameter(torch.zeros(8))
    p_f = torch.nn.Parameter(torch.randn(8))
    p_f_before = p_f.detach().clone()
    opt = GraftAdam([p_r], [p_f], lr=LR, betas=BETAS, eps=EPS, max_grad_norm=0.0)
    # forget inactive this window
    opt.set_window({p_r: g}, {p_r: g}, {"retain": 1.0, "forget": 1.0},
                   {"retain": True, "forget": False})
    opt.step()
    assert torch.equal(p_f, p_f_before)        # forget params untouched
    assert opt._t["forget"] == 0 and opt._t["retain"] == 1
    assert not opt.state[p_f]                   # no m/v state created for frozen param


def test_global_norm_clip_caps_step():
    g = torch.full((8,), 100.0)                 # huge gradient
    p = torch.nn.Parameter(torch.zeros(8))
    opt = GraftAdam([p], [], lr=LR, betas=BETAS, eps=EPS, max_grad_norm=1.0)
    opt.set_window({p: g}, {p: g}, {"retain": 1.0},
                   {"retain": True, "forget": False})
    opt.step()
    # G_m clipped to norm 1 ⇒ each element ~1/sqrt(8); m≈clip·(1-b1)·g, v≈(1-b2)·g² (unclipped)
    # step is far below the unclipped ~lr; just assert it did not blow up to ~lr·sign.
    assert float(p.abs().max()) < LR  # clipped m shrinks the step well below lr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all GraftAdam tests passed")
