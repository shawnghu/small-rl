"""Exp 4 (retain-hack PROHIBITION) tests.

1. EQUIVALENCE (the load-bearing claim): the fused decouple + PLAIN
   (single-moment) gradient accumulation produces adapter gradients EQUAL to a
   hand-rolled per-sample reference where each sample's retain param-grad is
   weighted by w_R and its forget param-grad by w_F, with the forward run at the
   sample's forget_fwd scale. The batch covers the prohibition masks w_R=-1,
   w_F in {1, 3} (plus the off-policy (0,-1,0) and a standard (1,1,1)), proving
   the fused fast path accepts arbitrary (incl. NEGATIVE and >1) mask constants
   with split_moment OFF and a balanced (per-sample) advantage. fp64, CPU.

   This generalizes tests/test_fused_routing_equivalence.py, whose stock
   reference only ever uses the gate masks {0, 1}. Here the reference scales the
   per-sample parameter gradient by an arbitrary weight via register_hook, which
   is exactly what _fused_decouple does per token (param grad = m * dg/dtheta,
   input grad preserved); under PLAIN Adam (no split-moment) the literal weight
   is a gradient-accumulation multiplier, matching the experiment's semantics.

2. CHARACTERIZATION: the per-sample triples (forget_fwd, retain_gm, forget_gm)
   derived from advantages.retain_prohibition_masks match the Exp 4 design for
   modes a/b/c.
"""

import os
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

from gradient_routing import (  # noqa: E402
    set_scales, set_fused_routing, clear_fused_routing,
)
from advantages import retain_prohibition_masks  # noqa: E402
from test_fused_routing_equivalence import (  # noqa: E402
    _build_mlp_tower, _build_lora_tower, _adapter_params, _seq_loss,
    _grouped_grads, _forward_tower, _compare,
)


# --------------------------------------------------------------------------- #
# Batch with per-sample (forget_fwd, w_R, w_F) triples covering the prohibition
# masks plus standard coherence / good cells.
# --------------------------------------------------------------------------- #

def _make_batch(hidden, seed):
    torch.manual_seed(seed)
    triples = [
        (1.0, -1.0, 1.0),   # mode (a) / mode (c) good : (1, -1, 1)
        (1.0, -1.0, 3.0),   # mode (c) bad             : (1, -1, 3)
        (0.0, -1.0, 0.0),   # mode (b)                 : (0, -1, 0) off-policy fwd
        (0.0,  1.0, 0.0),   # standard coherence       : (0,  1, 0)
        (1.0,  1.0, 1.0),   # standard good            : (1,  1, 1)
    ]
    lengths = [4, 3, 5, 2, 6]
    adv = [float(torch.randn(1).item()) for _ in triples]
    spans, off = [], 0
    for sid, L in enumerate(lengths):
        spans.append((off, off + L, sid))
        off += L
    T = off
    x = torch.randn(1, T, hidden, dtype=torch.float64, requires_grad=False)
    proj = torch.randn(hidden, dtype=torch.float64)
    return x, spans, triples, adv, proj


def _reference(tower, x, spans, triples, adv, proj, scale_denom):
    """Per-sample isolated forward+backward; scale retain param-grads by w_R and
    forget param-grads by w_F via register_hook; accumulate into .grad."""
    retain_params, forget_params = _adapter_params(tower)
    for p in retain_params + forget_params:
        p.grad = None
    for (a, b, sid) in spans:
        ffs, wR, wF = triples[sid]
        set_scales(tower, retain_scale=1.0, forget_scale=ffs)
        hooks = [p.register_hook(lambda g, w=wR: g * w) for p in retain_params]
        hooks += [p.register_hook(lambda g, w=wF: g * w) for p in forget_params]
        y = _forward_tower(tower, x[:, a:b, :])
        loss = _seq_loss(y, [(0, b - a, sid)], {sid: adv[sid]}, proj)
        (loss * (1.0 / scale_denom)).backward()
        for h in hooks:
            h.remove()
    set_scales(tower, retain_scale=1.0, forget_scale=1.0)
    return _grouped_grads(tower)


def _fused(tower, x, spans, triples, adv, proj, scale_denom):
    """One fused forward+backward over the whole heterogeneous pack, per-token
    (forget_fwd, retain_gm, forget_gm); PLAIN accumulation (no split-moment)."""
    retain_params, forget_params = _adapter_params(tower)
    for p in retain_params + forget_params:
        p.grad = None
    T = x.shape[1]
    ff = torch.zeros(T, dtype=torch.float64)
    rg = torch.zeros(T, dtype=torch.float64)
    fg = torch.zeros(T, dtype=torch.float64)
    advd = {}
    for (a, b, sid) in spans:
        ffs, wR, wF = triples[sid]
        ff[a:b], rg[a:b], fg[a:b] = ffs, wR, wF
        advd[sid] = adv[sid]
    set_scales(tower, retain_scale=1.0, forget_scale=1.0)  # forget_scale ignored when fused
    n_kept = len(spans)
    set_fused_routing(ff.view(1, T, 1), rg.view(1, T, 1), fg.view(1, T, 1))
    try:
        y = _forward_tower(tower, x)
        loss = _seq_loss(y, spans, advd, proj)
        (loss * (n_kept / scale_denom)).backward()
    finally:
        clear_fused_routing()
    return _grouped_grads(tower)


def _run_case(tower_fn, name):
    hidden = 8
    x, spans, triples, adv, proj = _make_batch(hidden, seed=11)
    scale_denom = len(spans) + 2  # > n_kept: emulate dropped samples
    tower = tower_fn()
    g_ref = _reference(tower, x, spans, triples, adv, proj, scale_denom)
    g_fused = _fused(tower, x, spans, triples, adv, proj, scale_denom)
    _compare(name, g_ref, g_fused)


def test_prohibition_equivalence_mlp():
    _run_case(lambda: _build_mlp_tower(3, 8, 4, seed=3), "MLP prohibition (w_R=-1, w_F in {1,3})")


def test_prohibition_equivalence_lora():
    _run_case(lambda: _build_lora_tower(3, 8, 2, seed=4), "LoRA prohibition (w_R=-1, w_F in {1,3})")


# --------------------------------------------------------------------------- #
# Characterization of the per-sample triples for modes a/b/c.
# --------------------------------------------------------------------------- #

def test_retain_prohibition_masks_constants():
    # (routing_forget_fwd_scale, rgm_good, fgm_good, rgm_bad, fgm_bad)
    assert retain_prohibition_masks("a") == (1.0, -1.0, 1.0, -1.0, 1.0)
    assert retain_prohibition_masks("b") == (0.0, -1.0, 0.0, -1.0, 0.0)
    assert retain_prohibition_masks("c") == (1.0, -1.0, 1.0, -1.0, 3.0)


def test_retain_prohibition_triples():
    # Derived per-sample TRIPLES (forget_fwd, retain_gm, forget_gm) good / bad.
    expected = {
        "a": ((1.0, -1.0, 1.0), (1.0, -1.0, 1.0)),
        "b": ((0.0, -1.0, 0.0), (0.0, -1.0, 0.0)),
        "c": ((1.0, -1.0, 1.0), (1.0, -1.0, 3.0)),
    }
    for mode, (good_exp, bad_exp) in expected.items():
        fs, rgm_good, fgm_good, rgm_bad, fgm_bad = retain_prohibition_masks(mode)
        assert (fs, rgm_good, fgm_good) == good_exp, (mode, "good")
        assert (fs, rgm_bad, fgm_bad) == bad_exp, (mode, "bad")


def test_retain_prohibition_masks_rejects_unknown():
    for bad in ("none", "d", "", "A"):
        with pytest.raises(ValueError):
            retain_prohibition_masks(bad)


if __name__ == "__main__":
    print("Exp 4 retain-prohibition tests:")
    test_prohibition_equivalence_mlp()
    test_prohibition_equivalence_lora()
    test_retain_prohibition_masks_constants()
    test_retain_prohibition_triples()
    test_retain_prohibition_masks_rejects_unknown()
    print("ALL PASS")
