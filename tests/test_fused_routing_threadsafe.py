"""Thread-safety test for fused gradient-routing under one-step off-policy.

Under one-step off-policy training the rollout thread forwards a frozen snapshot
view of the model CONCURRENTLY with the update thread's fused forward. The fused
per-token routing state (_FUSED_ROUTING in gradient_routing.py) is a process GLOBAL
so it stays visible to autograd's backward worker threads during gradient-checkpoint
recomputation (a thread-local would be lost there -> CheckpointError). To keep the
concurrent rollout thread from reading the update thread's masks (sized to a
different microbatch T -> broadcast shape crash), the rollout thread wraps its
no_grad forwards in `force_plain_forward()`, a thread-local opt-out that short-
circuits the adapter to the plain branch and never reads the global.

This test reproduces the race on a tiny CPU adapter:
  - an "update" thread repeatedly installs fused routing (global) with masks of
    length T_u and forwards a (1, T_u, H) batch,
  - a "rollout" thread, under force_plain_forward(), repeatedly forwards a
    (1, T_r, H) batch with T_r != T_u,
running both for many iterations. The rollout forward must always equal the
non-fused reference and neither thread may raise. A negative control (rollout NOT
wrapped in force_plain_forward) reproduces the broadcast crash.
"""

import os
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import (  # noqa: E402
    DualMLPAdapter,
    set_fused_routing,
    clear_fused_routing,
    force_plain_forward,
)


class _TinyMLP(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lin = torch.nn.Linear(h, h)

    def forward(self, x):
        return self.lin(x)


def _make_adapter(h=8, neurons=4):
    base = _TinyMLP(h)
    ad = DualMLPAdapter(base, hidden_size=h, retain_neurons=neurons,
                        forget_neurons=neurons)
    # Perturb the zero-init down projections so adapters actually contribute.
    with torch.no_grad():
        ad.down_retain.weight.normal_(0, 0.1)
        ad.down_forget.weight.normal_(0, 0.1)
    return ad


def _run_race(wrap_rollout_in_force_plain):
    """Returns (errors, rollout_mismatch). Empty/empty == passed."""
    torch.manual_seed(0)
    H = 8
    ad = _make_adapter(H)

    T_u, T_r = 70, 35719  # mismatched token counts (mirrors the real crash)
    x_r = torch.randn(1, T_r, H)
    with torch.no_grad():
        ref_r = ad(x_r)  # plain (non-fused) reference for the rollout batch

    errors = []
    rollout_mismatch = []
    stop = threading.Event()

    def update_loop():
        x_u = torch.randn(1, T_u, H)
        fwd_scale = torch.ones(1, T_u, 1)
        rmask = torch.ones(1, T_u, 1)
        fmask = torch.ones(1, T_u, 1)
        try:
            for _ in range(200):
                if stop.is_set():
                    break
                set_fused_routing(fwd_scale, rmask, fmask)
                try:
                    _ = ad(x_u)  # fused-path forward (global routing)
                finally:
                    clear_fused_routing()
        except Exception as e:  # pragma: no cover
            errors.append(("update", repr(e)))
            stop.set()

    def rollout_loop():
        try:
            for _ in range(200):
                if stop.is_set():
                    break
                with torch.no_grad():
                    if wrap_rollout_in_force_plain:
                        with force_plain_forward():
                            out = ad(x_r)
                    else:
                        out = ad(x_r)  # negative control: reads the global
                if not torch.allclose(out, ref_r, atol=1e-6):
                    rollout_mismatch.append(float((out - ref_r).abs().max()))
                    stop.set()
                    break
        except Exception as e:  # pragma: no cover
            errors.append(("rollout", repr(e)))
            stop.set()

    tu = threading.Thread(target=update_loop, name="update")
    tr = threading.Thread(target=rollout_loop, name="rollout")
    tu.start(); tr.start()
    tu.join(); tr.join()
    return errors, rollout_mismatch


def test_rollout_force_plain_isolates_from_global_routing():
    errors, rollout_mismatch = _run_race(wrap_rollout_in_force_plain=True)
    assert not errors, f"thread raised: {errors}"
    assert not rollout_mismatch, (
        f"rollout forward corrupted by update thread's routing: "
        f"max abs diff {rollout_mismatch}")


def test_negative_control_without_force_plain_crashes():
    # Without the opt-out, the rollout thread reads the global and hits the exact
    # production broadcast crash (or a value mismatch).
    errors, rollout_mismatch = _run_race(wrap_rollout_in_force_plain=False)
    assert errors or rollout_mismatch, (
        "expected the unguarded rollout to crash or mismatch, but it passed — "
        "the test no longer bites")


if __name__ == "__main__":
    test_rollout_force_plain_isolates_from_global_routing()
    test_negative_control_without_force_plain_crashes()
    print("ok: force_plain_forward isolates rollout; negative control bites")
