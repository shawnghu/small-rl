"""graft-port: SplitMomentAdamW participation factor + freeze + per-role step
counter + no-silent-fallback (split_moment.py). Pure-optimizer, CPU, synthetic grads.
"""
import torch
import pytest

from split_moment import SplitMomentAdamW


def _opt(roles_params, lr=0.1, beta2=0.999):
    groups = [{"params": [p], "lr": lr, "weight_decay": 0.0, "graft_role": role}
              for role, p in roles_params]
    return SplitMomentAdamW(groups, lr=lr, betas=(0.9, beta2), eps=1e-8, weight_decay=0.0)


def _set(p, g_m, g_v):
    p.grad = torch.full_like(p, g_m)
    p._pre_routing_grad = torch.full_like(p, g_v)


def test_participation_scales_step_by_inverse_c():
    # forget with c=2 takes ~1/2 the step of c=1 (v-source scaled -> sqrt(v) x2 -> step /2)
    p1 = torch.zeros(8, requires_grad=True)
    p2 = torch.zeros(8, requires_grad=True)
    o1, o2 = _opt([("forget", p1)]), _opt([("forget", p2)])
    for _ in range(80):
        _set(p1, 1.0, 1.0); _set(p2, 1.0, 1.0)
        o1.set_window({"forget": 1.0}, {"forget": True}); o1.step()
        o2.set_window({"forget": 2.0}, {"forget": True}); o2.step()
    ratio = p2.abs().mean().item() / p1.abs().mean().item()
    assert abs(ratio - 0.5) < 0.02, ratio


def test_freeze_leaves_param_and_state_untouched():
    p = torch.zeros(4, requires_grad=True)
    o = _opt([("forget", p)])
    _set(p, 1.0, 1.0)
    o.set_window({"forget": 1.0}, {"forget": True}); o.step()
    after1 = p.clone()
    assert o.state[p]["step"] == 1
    _set(p, 5.0, 5.0)                                   # big grads, but frozen
    o.set_window({"forget": 1.0}, {"forget": False}); o.step()
    assert torch.equal(p, after1), "frozen forget param moved"
    assert o.state[p]["step"] == 1, "frozen window advanced the step counter"


def test_per_role_step_counter_lags_on_frozen_windows():
    pr = torch.zeros(4, requires_grad=True)
    pf = torch.zeros(4, requires_grad=True)
    o = _opt([("retain", pr), ("forget", pf)])
    for active_f in (True, False, True, False, True):
        _set(pr, 1.0, 1.0); _set(pf, 1.0, 1.0)
        o.set_window({"retain": 1.0, "forget": 1.0},
                     {"retain": True, "forget": active_f})
        o.step()
    assert o.state[pr]["step"] == 5        # retain active every window
    assert o.state[pf]["step"] == 3        # forget active on 3 of 5 -> per-role t


def test_no_silent_fallback_for_routing_param():
    p = torch.zeros(4, requires_grad=True)
    p.grad = torch.ones(4)
    p._pre_routing_grad = None             # capture missing on a tagged (routing) param
    o = _opt([("forget", p)])
    o.set_window({"forget": 1.0}, {"forget": True})
    with pytest.raises(AssertionError):
        o.step()


def test_untagged_param_keeps_adamw_fallback():
    # no graft_role -> g_v None falls back to g_m (plain AdamW), no raise
    p = torch.zeros(4, requires_grad=True)
    p.grad = torch.ones(4)
    o = SplitMomentAdamW([{"params": [p], "lr": 0.1, "weight_decay": 0.0}],
                         lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    o.step()                               # must not raise
    assert not torch.equal(p, torch.zeros(4))


def test_window_c1_matches_no_window():
    pa = torch.zeros(4, requires_grad=True)
    pb = torch.zeros(4, requires_grad=True)
    oa, ob = _opt([("forget", pa)]), _opt([("forget", pb)])
    for _ in range(10):
        _set(pa, 0.7, 1.3); _set(pb, 0.7, 1.3)
        oa.set_window({"forget": 1.0}, {"forget": True}); oa.step()
        ob.step()                          # no window -> default c=1/active
    assert torch.allclose(pa, pb, atol=1e-7)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
