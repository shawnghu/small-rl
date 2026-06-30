"""Exp 5 invariants: retain-only weight decay + ema_clamp forget-scale clamp.

(1) Optimizer param-group construction: --weight_decay applies ONLY to the
    retain adapter; the forget group is always weight_decay=0.0. Tested for both
    optimizer paths that build retain/forget groups (split_moment=on via
    with_role=True, asymmetric-LR / split_moment=off via with_role=False),
    plus a behavioral check that SplitMomentAdamW actually decays retain but NOT
    forget on an ACTIVE (non-frozen) window when fed groups built this way.

(2) ema_clamp clamp dynamics: the pure clamp-decay recurrence is monotone
    non-increasing and floored at forget_scale_min_clamp.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train  # noqa: E402
from split_moment import SplitMomentAdamW  # noqa: E402

_build = train.SampleGRPOTrainer._build_retain_forget_groups
_decay = train.SampleGRPOTrainer._decay_forget_scale_clamp


# ---------------------------------------------------------------------------
# (1) param-group weight-decay policy
# ---------------------------------------------------------------------------

def _groups(with_role, wd=1.0, lr=3e-4, forget_lr_mult=1.0):
    r = torch.nn.Parameter(torch.zeros(2))
    f = torch.nn.Parameter(torch.zeros(2))
    return _build([r], [f], lr=lr, weight_decay=wd,
                  forget_lr_mult=forget_lr_mult, with_role=with_role)


def test_retain_decays_forget_does_not_split_on():
    # split_moment path (with_role=True): role-tagged groups.
    g = _groups(with_role=True, wd=1.0)
    retain, forget = g
    assert retain["weight_decay"] == 1.0, retain
    assert forget["weight_decay"] == 0.0, forget
    assert retain["graft_role"] == "retain"
    assert forget["graft_role"] == "forget"


def test_retain_decays_forget_does_not_split_off():
    # asymmetric-LR / non-split path (with_role=False): no graft_role tag.
    g = _groups(with_role=False, wd=1.0)
    retain, forget = g
    assert retain["weight_decay"] == 1.0, retain
    assert forget["weight_decay"] == 0.0, forget
    assert "graft_role" not in retain and "graft_role" not in forget


def test_forget_wd_zero_for_any_wd():
    for wd in (0.0, 0.01, 1.0, 3.0, 10.0):
        for with_role in (True, False):
            _, forget = _groups(with_role=with_role, wd=wd)
            assert forget["weight_decay"] == 0.0, (wd, with_role)


def test_forget_lr_scaled_by_mult():
    retain, forget = _groups(with_role=True, lr=2e-4, forget_lr_mult=0.5)
    assert math.isclose(retain["lr"], 2e-4)
    assert math.isclose(forget["lr"], 1e-4)


def test_split_moment_decays_retain_not_forget_on_active_window():
    """End-to-end: build the exact groups train.py builds and run one
    SplitMomentAdamW step on an ACTIVE window (both roles participate, no
    freeze). With ZERO gradient, the only thing that can move a param is
    decoupled weight decay. The retain param must shrink (wd=1.0); the forget
    param must be untouched (wd=0.0)."""
    wd, lr = 1.0, 1e-2
    rp = torch.nn.Parameter(torch.full((4,), 2.0))
    fp = torch.nn.Parameter(torch.full((4,), 2.0))
    groups = _build([rp], [fp], lr=lr, weight_decay=wd,
                    forget_lr_mult=1.0, with_role=True)
    opt = SplitMomentAdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=wd)
    # Zero grads (decay-only probe), with the pre-routing capture present so the
    # no-fallback assert is satisfied for the role-tagged params.
    for p in (rp, fp):
        p.grad = torch.zeros_like(p)
        p._pre_routing_grad = torch.zeros_like(p)
    # ACTIVE window: both roles participate -> no freeze, so wd (if any) bites.
    opt.set_window(participation={"retain": 1.0, "forget": 1.0},
                   active={"retain": True, "forget": True})
    opt.step()
    # retain shrank by exactly (1 - lr*wd); forget unchanged.
    torch.testing.assert_close(rp.detach(), torch.full((4,), 2.0 * (1 - lr * wd)),
                               rtol=0, atol=1e-7)
    torch.testing.assert_close(fp.detach(), torch.full((4,), 2.0),
                               rtol=0, atol=0)


# ---------------------------------------------------------------------------
# (2) ema_clamp clamp dynamics
# ---------------------------------------------------------------------------

def test_clamp_monotone_non_increasing_and_floored():
    decay, floor = 0.9, 0.1
    clamp = 1.0
    seq = [clamp]
    for _ in range(100):
        clamp = _decay(clamp, decay, floor)
        seq.append(clamp)
    # monotone non-increasing
    for a, b in zip(seq, seq[1:]):
        assert b <= a + 1e-12, (a, b)
    # floored
    assert min(seq) >= floor - 1e-12
    assert math.isclose(seq[-1], floor, abs_tol=1e-9), seq[-1]
    # first step is exactly one multiply
    assert math.isclose(seq[1], 1.0 * decay)


def test_clamp_floor_zero_decays_toward_zero():
    clamp = 1.0
    for _ in range(200):
        clamp = _decay(clamp, 0.9, 0.0)
    assert clamp < 1e-6


if __name__ == "__main__":
    test_retain_decays_forget_does_not_split_on()
    test_retain_decays_forget_does_not_split_off()
    test_forget_wd_zero_for_any_wd()
    test_forget_lr_scaled_by_mult()
    test_split_moment_decays_retain_not_forget_on_active_window()
    test_clamp_monotone_non_increasing_and_floored()
    test_clamp_floor_zero_decays_toward_zero()
    print("PASS ✓")
