"""Thread-safety test for fused gradient-routing state under one-step off-policy.

Under one-step off-policy training the rollout thread forwards a frozen snapshot
view of the model CONCURRENTLY with the update thread's fused forward. The fused
per-token routing state (_FUSED_ROUTING_TLS in gradient_routing.py) must be
thread-local: if it were a shared process global, the rollout thread's plain
forward would read the masks the update thread installed for a *different*
microbatch (different token count T) and either crash on a broadcast shape
mismatch or silently corrupt the forward value.

This test reproduces that exact race on tiny CPU adapters:
  - an "update" thread repeatedly installs fused routing with masks of length T_u
    and forwards a (1, T_u, H) batch,
  - a "rollout" thread repeatedly does a PLAIN forward (no fused routing) over a
    (1, T_r, H) batch with T_r != T_u,
running both for many iterations. With thread-local state the rollout forward
always equals the non-fused reference and neither thread raises; a shared global
would raise a RuntimeError (shape mismatch) or produce a wrong value.
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


def test_fused_routing_is_thread_local():
    torch.manual_seed(0)
    H = 8
    ad = _make_adapter(H)

    T_u, T_r = 70, 35719  # mismatched token counts (mirrors the real crash)
    x_r = torch.randn(1, T_r, H)

    # Reference: plain (non-fused) forward value for the rollout batch.
    with torch.no_grad():
        ref_r = ad(x_r)

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
                    _ = ad(x_u)  # fused-path forward on this thread
                finally:
                    clear_fused_routing()
        except Exception as e:  # pragma: no cover - failure path
            errors.append(("update", repr(e)))
            stop.set()

    def rollout_loop():
        try:
            for _ in range(200):
                if stop.is_set():
                    break
                with torch.no_grad():
                    out = ad(x_r)  # MUST take the plain path on this thread
                if not torch.allclose(out, ref_r, atol=1e-6):
                    rollout_mismatch.append(float((out - ref_r).abs().max()))
                    stop.set()
                    break
        except Exception as e:  # pragma: no cover - failure path
            errors.append(("rollout", repr(e)))
            stop.set()

    tu = threading.Thread(target=update_loop, name="update")
    tr = threading.Thread(target=rollout_loop, name="rollout")
    tu.start(); tr.start()
    tu.join(); tr.join()

    assert not errors, f"thread raised (shared-global race): {errors}"
    assert not rollout_mismatch, (
        f"rollout forward corrupted by update thread's routing: "
        f"max abs diff {rollout_mismatch}")


if __name__ == "__main__":
    test_fused_routing_is_thread_local()
    print("ok: fused routing state is thread-local")
