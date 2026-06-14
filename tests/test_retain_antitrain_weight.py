"""Correctness test for the retain anti-train weight W (routed/bad samples).

The feature: on routed (bad) samples, scale the RETAIN adapter's *parameter*
gradient by a weight W instead of zeroing it (classic routing = W=0). The forget
adapter is unchanged. Implemented by setting the fused per-token retain_grad_mask
to W on bad-sample tokens (train.py:_fused_forward_backward), which _fused_decouple
multiplies into the retain param gradient.

Two mathematical properties are checked here, reusing the tiny-CPU-tower harness
from test_fused_routing_equivalence:

1. Linearity in W. Because the total retain param gradient is a sum of per-sample
   contributions each weighted by its retain_grad_mask, and only bad samples'
   mask changes with W (coh/good stay at 1.0):
       retain_grad(W) == retain_grad(0) + W * (retain_grad(1) - retain_grad(0))
   The bracket is exactly "the gradient bad samples would contribute to retain if
   trained fully" — so scaling it by W is identical to scaling the bad-sample
   retain advantage by W. W=0 reproduces classic routing (no retain update on bad).

2. W-invariance of the forget adapter. The forget mask never depends on W, so
   forget param gradients are bit-identical across all W.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import (  # noqa: E402
    set_scales, set_fused_routing, clear_fused_routing,
)
from tests.test_fused_routing_equivalence import (  # noqa: E402
    _build_mlp_tower, _build_lora_tower, _make_batch,
    _forward_tower, _adapter_params, _seq_loss,
)


def _fused_with_W(tower, x_full, spans_full, classes, adv, proj,
                  train_fs, scale_denom, W):
    """Single fused pass; bad-sample retain mask = W (classic routing is W=0).
    Returns (retain_grads, forget_grads) as detached clones in param order."""
    retain_params, forget_params = _adapter_params(tower)
    for p in retain_params + forget_params:
        p.grad = None

    T = x_full.shape[1]
    forget_fwd = torch.zeros(T, dtype=torch.float64)
    retain_gm = torch.zeros(T, dtype=torch.float64)
    forget_gm = torch.zeros(T, dtype=torch.float64)
    for (a, b, sid) in spans_full:
        cls = classes[sid]
        if cls == "coherence":
            ffs, rgm, fgm = 0.0, 1.0, 0.0
        elif cls == "good":
            ffs, rgm, fgm = train_fs, 1.0, 1.0  # classic (non-exclusive)
        else:  # bad: retain anti-trained by W; forget trains normally
            ffs, rgm, fgm = train_fs, W, 1.0
        forget_fwd[a:b] = ffs
        retain_gm[a:b] = rgm
        forget_gm[a:b] = fgm

    set_scales(tower, retain_scale=1.0, forget_scale=1.0)
    n_kept = len(spans_full)
    set_fused_routing(forget_fwd.view(1, T, 1), retain_gm.view(1, T, 1),
                      forget_gm.view(1, T, 1))
    try:
        y = _forward_tower(tower, x_full)
        loss = _seq_loss(y, spans_full, adv, proj)
        (loss * (n_kept / scale_denom)).backward()
    finally:
        clear_fused_routing()

    rg = [None if p.grad is None else p.grad.detach().clone() for p in retain_params]
    fg = [None if p.grad is None else p.grad.detach().clone() for p in forget_params]
    return rg, fg


def _run_case(tower_fn, name):
    hidden = 8
    x, spans, classes, adv, proj = _make_batch(hidden, seed=7)
    train_fs = 0.7
    scale_denom = len(spans) + 2

    tower = tower_fn()
    r0, f0 = _fused_with_W(tower, x, spans, classes, adv, proj, train_fs, scale_denom, W=0.0)
    r1, f1 = _fused_with_W(tower, x, spans, classes, adv, proj, train_fs, scale_denom, W=1.0)

    worst_lin = 0.0
    worst_forget = 0.0
    for W in (-1.0, -0.5, -0.1, 0.3):
        rW, fW = _fused_with_W(tower, x, spans, classes, adv, proj, train_fs, scale_denom, W=W)
        for g0, g1, gW in zip(r0, r1, rW):
            if g0 is None and g1 is None and gW is None:
                continue
            pred = g0 + W * (g1 - g0)          # linearity in W
            denom = pred.abs().max().item()
            diff = (pred - gW).abs().max().item()
            worst_lin = max(worst_lin, diff / denom if denom > 0 else diff)
        for ga, gb in zip(f0, fW):              # forget invariant to W
            if ga is None and gb is None:
                continue
            denom = ga.abs().max().item()
            diff = (ga - gb).abs().max().item()
            worst_forget = max(worst_forget, diff / denom if denom > 0 else diff)

    print(f"  {name}: worst retain-linearity rel = {worst_lin:.3e}, "
          f"worst forget-invariance rel = {worst_forget:.3e}")
    assert worst_lin < 1e-9, f"{name}: retain_grad(W) not linear in W ({worst_lin:.3e})"
    assert worst_forget < 1e-12, f"{name}: forget grad varied with W ({worst_forget:.3e})"

    # Sanity: at W=0 the retain grad is exactly classic routing (bad samples
    # contribute nothing), and a nonzero bad contribution actually exists, so the
    # test bites rather than passing trivially.
    bad_contrib = max((g1 - g0).abs().max().item() for g0, g1 in zip(r0, r1))
    assert bad_contrib > 1e-6, f"{name}: bad-sample retain contribution ~0; test is vacuous"


def test_mlp_antitrain_weight():
    _run_case(lambda: _build_mlp_tower(3, 8, 4, seed=1), "MLP")


def test_lora_antitrain_weight():
    _run_case(lambda: _build_lora_tower(3, 8, 2, seed=2), "LoRA")


if __name__ == "__main__":
    print("Retain anti-train weight W (linearity + forget-invariance):")
    test_mlp_antitrain_weight()
    test_lora_antitrain_weight()
    print("ALL PASS ✓")
