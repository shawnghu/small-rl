"""Unit tests for SplitMomentAdamW.

Pins two properties:
  (1) When the pre-routing gradient equals .grad, SplitMomentAdamW reproduces
      torch.optim.AdamW bit-for-bit (it's a strict generalization).
  (2) The second moment v is driven by ._pre_routing_grad, not .grad — verified
      by hand against the closed-form first step.
"""

import math
import os
import sys

import torch
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from split_moment import SplitMomentAdamW, clip_pre_routing_grads_  # noqa: E402


def _run(opt_cls, *, split, steps=6, seed=0):
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(5, 4))
    opt = opt_cls([p], lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    torch.manual_seed(seed + 1)
    for _ in range(steps):
        opt.zero_grad()
        g = torch.randn(5, 4)
        p.grad = g.clone()
        if split:
            p._pre_routing_grad = g.clone()  # pre == post -> must equal AdamW
        opt.step()
    return p.detach().clone()


def test_matches_adamw_when_pre_equals_post():
    a = _run(AdamW, split=False)
    b = _run(SplitMomentAdamW, split=True)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_v_uses_pre_routing_grad():
    # One step, no weight decay: closed form is checkable by hand.
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(3, 3))
    p0 = p.detach().clone()
    lr, b1, b2, eps = 1e-2, 0.9, 0.999, 1e-8
    opt = SplitMomentAdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0)
    g_m = torch.randn(3, 3)          # routed grad -> first moment
    g_v = torch.randn(3, 3) * 5.0    # pre-routing grad -> second moment (different)
    opt.zero_grad()
    p.grad = g_m.clone()
    p._pre_routing_grad = g_v.clone()
    opt.step()

    # Hand-computed AdamW step with m<-g_m, v<-g_v:
    m = (1 - b1) * g_m
    v = (1 - b2) * (g_v * g_v)
    mhat = m / (1 - b1)
    vhat = v / (1 - b2)
    expected = p0 - lr * mhat / (vhat.sqrt() + eps)
    torch.testing.assert_close(p.detach(), expected, rtol=0, atol=1e-6)

    # And it must DIFFER from the all-g_m optimizer (v from g_m), proving v used g_v.
    torch.manual_seed(3)
    p2 = torch.nn.Parameter(p0.clone())
    opt2 = SplitMomentAdamW([p2], lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0)
    p2.grad = g_m.clone()
    p2._pre_routing_grad = g_m.clone()
    opt2.step()
    assert (p2.detach() - p.detach()).abs().max() > 1e-5


def test_clip_matches_dotgrad_clip():
    # The pre-routing grad (v) must be scaled by the EXACT same coefficient that
    # torch.nn.utils.clip_grad_norm_ applies to .grad (m), so clipping is one
    # shared event across both moments.
    from torch.nn.utils import clip_grad_norm_
    torch.manual_seed(5)
    p = torch.nn.Parameter(torch.randn(10, 10))
    g_post = torch.randn(10, 10) * 4.0   # large norm -> clip bites
    g_pre = torch.randn(10, 10) * 4.0    # different gradient for v
    p.grad = g_post.clone()
    p._pre_routing_grad = g_pre.clone()
    max_norm = 1.0

    before = p.grad.clone()
    total_norm = clip_grad_norm_([p], max_norm)         # clips .grad in place
    applied_to_grad = (p.grad / before).flatten()[0].item()
    assert applied_to_grad < 1.0, "clip did not engage; pick a larger grad"

    coef = clip_pre_routing_grads_([{"params": [p]}], max_norm, total_norm)
    # recomputed coef == coef torch applied to .grad
    assert abs(coef - applied_to_grad) < 1e-6, (coef, applied_to_grad)
    torch.testing.assert_close(p._pre_routing_grad, g_pre * applied_to_grad,
                               rtol=0, atol=1e-6)


def test_clip_noop_when_under_threshold():
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.full((4,), 0.01)
    p._pre_routing_grad = torch.full((4,), 0.01)
    pre = p._pre_routing_grad.clone()
    # tiny norm, max_norm large -> coef would be >1 -> no scaling
    coef = clip_pre_routing_grads_([{"params": [p]}], max_norm=1.0,
                                   total_norm=torch.tensor(0.02))
    assert coef is None
    torch.testing.assert_close(p._pre_routing_grad, pre, rtol=0, atol=0)


if __name__ == "__main__":
    test_matches_adamw_when_pre_equals_post()
    test_v_uses_pre_routing_grad()
    test_clip_matches_dotgrad_clip()
    test_clip_noop_when_under_threshold()
    print("PASS ✓")
