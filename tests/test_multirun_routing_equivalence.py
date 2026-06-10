"""Exact-equivalence test for multi-run fused gradient routing.

The claim: ONE shared base forward + ONE backward over a packed batch mixing
sequences from N independent runs (each run = its own retain+forget adapter
pair on the SAME frozen base, via MultiRunDualAdapter + per-run per-token
routing tensors) produces, for every run i:

  1. forward values on run i's tokens identical to a single-run fused model
     holding only run i's adapters (the run forward-mask keeps each run's
     residual stream clean), and
  2. adapter gradients identical to running the single-run fused step over
     ONLY run i's sequences (cross-run isolation + correct per-run
     retain/forget routing + per-run loss weighting).

Reference per run = the validated single-run fused step from
tests/test_fused_routing_equivalence.py, run on that run's slice of the batch.
CPU, float64, tiny towers — exact to ~1e-9 (observed ~1e-15).
"""

import copy
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_fused_routing_equivalence as single  # noqa: E402
from gradient_routing import (  # noqa: E402
    DualLoRALinear, DualMLPAdapter, MultiRunDualAdapter,
    collect_routing_params_per_run,
    set_fused_routing, clear_fused_routing,
    set_fused_routing_multi, clear_fused_routing_multi,
)


# --------------------------------------------------------------------------- #
# Tower construction: shared frozen bases, N single towers, one multi tower
# --------------------------------------------------------------------------- #

def _build_setup(kind, n_runs, n_layers, hidden, width, base_seed, run_seed0):
    """Returns (singles, multi).

    singles[r]: ModuleList of vanilla adapters over value-identical base copies
    (run r's adapters seeded run_seed0+r).
    multi: ModuleList of MultiRunDualAdapter sharing ONE base per layer, whose
    children are weight-identical deepcopies of singles[r]'s adapters.
    """
    torch.manual_seed(base_seed)
    bases = [nn.Linear(hidden, hidden, bias=False).double() for _ in range(n_layers)]
    for b in bases:
        for p in b.parameters():
            p.requires_grad = False

    base_attr = "base_mlp" if kind == "mlp" else "base_layer"
    singles = []
    for r in range(n_runs):
        torch.manual_seed(run_seed0 + r)
        layers = []
        for li in range(n_layers):
            if kind == "mlp":
                lay = DualMLPAdapter(copy.deepcopy(bases[li]), hidden, width, width).double()
            else:
                lay = DualLoRALinear(copy.deepcopy(bases[li]), rank=width, forget_rank=width,
                                     alpha=16, dropout=0.0).double()
            layers.append(lay)
        tower = nn.ModuleList(layers)
        single._perturb_down(tower)  # consumes the run-seeded RNG -> per-run weights
        singles.append(tower)

    multi_layers = []
    for li in range(n_layers):
        children = []
        for r in range(n_runs):
            child = copy.deepcopy(singles[r][li])
            child._modules[base_attr] = bases[li]  # share the wrapper's base
            children.append(child)
        multi_layers.append(MultiRunDualAdapter(bases[li], children))
    return singles, nn.ModuleList(multi_layers)


# --------------------------------------------------------------------------- #
# Mixed-run packed batch
# --------------------------------------------------------------------------- #

_CLS_CYCLE = [("coherence", 3), ("good", 4), ("bad", 6),
              ("good", 2), ("bad", 5), ("coherence", 4)]


def _make_batch(hidden, n_runs, seed):
    rng = random.Random(seed)
    seqs = []
    sid = 0
    for r in range(n_runs):
        for cls, L in _CLS_CYCLE:
            seqs.append((sid, r, cls, L + (r % 3)))  # vary lengths by run
            sid += 1
    rng.shuffle(seqs)  # interleave runs' sequences in pack order

    spans, run_of, cls_of, off = [], {}, {}, 0
    for (s, r, cls, L) in seqs:
        spans.append((off, off + L, s))
        run_of[s], cls_of[s] = r, cls
        off += L
    T = off
    torch.manual_seed(seed)
    x = torch.randn(1, T, hidden, dtype=torch.float64)
    adv = {s: float(torch.randn(1).item()) for (s, _, _, _) in seqs}
    proj = torch.randn(hidden, dtype=torch.float64)
    return x, spans, run_of, cls_of, adv, proj


def _run_slice(x, spans, run_of, r):
    """Run r's sequences: sliced x (pack order preserved) + rebased spans."""
    spans_r = [(a, b, s) for (a, b, s) in spans if run_of[s] == r]
    x_r = torch.cat([x[:, a:b] for (a, b, _) in spans_r], dim=1)
    rebased, off = [], 0
    for (a, b, s) in spans_r:
        rebased.append((off, off + (b - a), s))
        off += (b - a)
    return x_r, spans_r, rebased


def _class_masks(cls, train_fs, exclusive):
    """(forget_fwd_scale, retain_grad, forget_grad) for one token class —
    identical to the conventions in the single-run fused step/test."""
    if cls == "coherence":
        return 0.0, 1.0, 0.0
    if cls == "good":
        return train_fs, 1.0, (0.0 if exclusive else 1.0)
    return train_fs, 0.0, 1.0  # bad


def _build_multi_masks(spans, run_of, cls_of, n_runs, T, train_fs, exclusive):
    # (n_runs, 1, T, 1): row i broadcasts against the packed (1, T, H) activation.
    ffs = torch.zeros(n_runs, 1, T, 1, dtype=torch.float64)
    rfm = torch.zeros(n_runs, 1, T, 1, dtype=torch.float64)
    rgm = torch.zeros(n_runs, 1, T, 1, dtype=torch.float64)
    fgm = torch.zeros(n_runs, 1, T, 1, dtype=torch.float64)
    for (a, b, s) in spans:
        r = run_of[s]
        f, rg, fg = _class_masks(cls_of[s], train_fs, exclusive)
        rfm[r, 0, a:b] = 1.0
        ffs[r, 0, a:b] = f
        rgm[r, 0, a:b] = rg
        fgm[r, 0, a:b] = fg
    return ffs, rfm, rgm, fgm


# --------------------------------------------------------------------------- #
# Single-run reference forward value (for the residual-stream check)
# --------------------------------------------------------------------------- #

def _single_forward_value(tower, x_r, spans_rb, cls_of, train_fs, exclusive):
    T = x_r.shape[1]
    ffs = torch.zeros(T, dtype=torch.float64)
    rgm = torch.zeros(T, dtype=torch.float64)
    fgm = torch.zeros(T, dtype=torch.float64)
    for (a, b, s) in spans_rb:
        f, rg, fg = _class_masks(cls_of[s], train_fs, exclusive)
        ffs[a:b], rgm[a:b], fgm[a:b] = f, rg, fg
    set_fused_routing(ffs.view(1, T, 1), rgm.view(1, T, 1), fgm.view(1, T, 1))
    try:
        with torch.no_grad():
            y = single._forward_tower(tower, x_r)
    finally:
        clear_fused_routing()
    return y


# --------------------------------------------------------------------------- #
# Grad pairing across towers (params differ by object; pair by position)
# --------------------------------------------------------------------------- #

def _adapter_params(layers):
    out = []
    for lay in layers:
        out += lay.get_retain_params() + lay.get_forget_params()
    return out


def _compare_grads(name, params_ref, params_test):
    assert len(params_ref) == len(params_test)
    worst = 0.0
    for p_r, p_t in zip(params_ref, params_test):
        g_r = p_r.grad if p_r.grad is not None else torch.zeros_like(p_r)
        g_t = p_t.grad if p_t.grad is not None else torch.zeros_like(p_t)
        denom = g_r.abs().max().item()
        diff = (g_r - g_t).abs().max().item()
        worst = max(worst, diff / denom if denom > 0 else diff)
    print(f"  {name}: worst relative grad diff = {worst:.3e}")
    assert worst < 1e-9, f"{name}: multi-run != per-run single reference ({worst:.3e})"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _run_case(kind, exclusive, n_runs=3, n_layers=3, hidden=8, width=4):
    train_fs = 0.7
    singles, multi = _build_setup(kind, n_runs, n_layers, hidden, width,
                                  base_seed=11, run_seed0=100)
    x, spans, run_of, cls_of, adv, proj = _make_batch(hidden, n_runs, seed=7)
    T = x.shape[1]
    # Unequal per-run loss denominators to exercise per-run weighting.
    denom = {r: len([1 for s in run_of if run_of[s] == r]) + 2 + r
             for r in range(n_runs)}

    # --- Reference: per run, the validated single-run fused step on its slice.
    for r in range(n_runs):
        x_r, spans_r, rebased = _run_slice(x, spans, run_of, r)
        cls_r = {s: cls_of[s] for (_, _, s) in spans_r}
        single._fused(singles[r], x_r, rebased, cls_r, adv, proj,
                      train_fs, exclusive, denom[r])

    # --- Multi-run: one shared forward, one summed loss, one backward.
    for lay in multi:
        for p in lay.get_retain_params() + lay.get_forget_params():
            p.grad = None
    ffs, rfm, rgm, fgm = _build_multi_masks(spans, run_of, cls_of, n_runs, T,
                                            train_fs, exclusive)
    set_fused_routing_multi(ffs, rfm, rgm, fgm)
    try:
        y = single._forward_tower(multi, x)
        loss = torch.zeros((), dtype=torch.float64)
        for r in range(n_runs):
            spans_r = [(a, b, s) for (a, b, s) in spans if run_of[s] == r]
            loss = loss + single._seq_loss(y, spans_r, adv, proj) \
                * (len(spans_r) / denom[r])
        loss.backward()
    finally:
        clear_fused_routing_multi()

    # --- 1) Residual-stream check: multi output on run r's tokens == single-run
    #        fused forward of tower r on its slice.
    with torch.no_grad():
        for r in range(n_runs):
            x_r, spans_r, rebased = _run_slice(x, spans, run_of, r)
            y_r = _single_forward_value(singles[r], x_r, rebased, cls_of,
                                        train_fs, exclusive)
            off = 0
            worst = 0.0
            for (a, b, _) in spans_r:
                seg = (y[0, a:b] - y_r[0, off:off + (b - a)]).abs().max().item()
                worst = max(worst, seg)
                off += (b - a)
            assert worst < 1e-12, (
                f"{kind} run {r}: forward residual stream diverges ({worst:.3e}) — "
                f"run forward-mask is leaking foreign adapter contributions")

    # --- 2) Per-run gradient equivalence.
    for r in range(n_runs):
        _compare_grads(
            f"{kind} (exclusive={exclusive}) run {r}",
            _adapter_params(singles[r]),
            _adapter_params([w.runs[r] for w in multi]),
        )

    # --- 3) Per-run param collection sanity.
    class _M(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers
    per_run = collect_routing_params_per_run(_M(multi))
    assert len(per_run) == n_runs
    for r, (ret, forg) in enumerate(per_run):
        assert len(ret) + len(forg) == len(_adapter_params([w.runs[r] for w in multi]))


def test_state_dict_base_once():
    """The shared frozen base must serialize exactly once per wrapper."""
    singles, multi = _build_setup("mlp", 3, 2, 8, 4, base_seed=1, run_seed0=50)
    sd = multi[0].state_dict()
    base_keys = [k for k in sd if k.startswith("base.")]
    child_base_keys = [k for k in sd if "base_mlp" in k or "base_layer" in k]
    assert base_keys and not child_base_keys, (
        f"base serialization wrong: base_keys={base_keys}, "
        f"child_base_keys={child_base_keys}")


def test_stateless_forward_guard():
    """No multi state: base-only iff all scales zero; loud assert otherwise."""
    _, multi = _build_setup("mlp", 2, 1, 8, 4, base_seed=2, run_seed0=60)
    w = multi[0]
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    try:
        w(x)
        raise RuntimeError("expected AssertionError without multi routing state")
    except AssertionError:
        pass
    for c in w.runs:
        c.retain_scale = 0.0
        c.forget_scale = 0.0
    with torch.no_grad():
        y = w(x)
        assert torch.allclose(y, w.base(x)), "scales-zero forward must be base-only"


def test_mlp_classic():
    _run_case("mlp", exclusive=False)


def test_mlp_exclusive():
    _run_case("mlp", exclusive=True)


def test_lora_classic():
    _run_case("lora", exclusive=False)


def test_lora_exclusive():
    _run_case("lora", exclusive=True)


if __name__ == "__main__":
    print("Multi-run fused routing equivalence (N-run shared-base step vs N single-run fused steps):")
    test_state_dict_base_once()
    print("  state_dict: base serialized once ✓")
    test_stateless_forward_guard()
    print("  stateless-forward guard ✓")
    test_mlp_classic()
    test_mlp_exclusive()
    test_lora_classic()
    test_lora_exclusive()
    print("ALL PASS ✓")
