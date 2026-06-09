"""Exact-equivalence test for the fused gradient-routing reduction.

Validates the novel mechanism in gradient_routing.py (per-token forward
forget-scale + activation-grad masks on retain_out/forget_out) independently of
the liger/packed loss path, on tiny CPU towers in float64.

The claim: ONE forward+backward over a heterogeneous packed batch (coherence +
good + bad samples), with per-token routing, produces the SAME adapter gradients
as the stock homogeneous-microbatch scheme — separate per-class forward+backward
passes with scalar forget-scales and parameter register_hooks — under a
per-sequence-normalized GRPO-like loss. (Per-sequence normalization is additive
across sequences, so the per-class scale n_mb/scale_denom sums to the fused
n_kept/scale_denom.)

This is the same equivalence bench_fused_gr.py checks end-to-end on GPU against
the real liger loss; here we isolate the reduction so it can run without CUDA.
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import (  # noqa: E402
    DualLoRALinear, DualMLPAdapter, set_scales,
    set_fused_routing, clear_fused_routing,
)


# --------------------------------------------------------------------------- #
# Tiny packed tower
# --------------------------------------------------------------------------- #

def _perturb_down(tower):
    """Adapters zero-init their output projection (down / lora_B), so a fresh
    tower's adapter outputs — and the gate/up gradients — are all zero. Give the
    output projections small random weights so every adapter parameter has a
    non-trivial gradient and the equivalence test actually bites."""
    with torch.no_grad():
        for layer in tower:
            for attr in ("down_retain", "down_forget"):
                m = getattr(layer, attr, None)
                if m is not None:
                    m.weight.normal_(0, 0.1)
            for attr in ("lora_B_retain", "lora_B_forget"):
                p = getattr(layer, attr, None)
                if p is not None:
                    p.normal_(0, 0.1)
    return tower


def _build_mlp_tower(n_layers, hidden, neurons, seed):
    torch.manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        base = nn.Linear(hidden, hidden, bias=False).double()
        layers.append(DualMLPAdapter(base, hidden, neurons, neurons).double())
    return _perturb_down(nn.ModuleList(layers))


def _build_lora_tower(n_layers, hidden, rank, seed):
    torch.manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        base = nn.Linear(hidden, hidden, bias=False).double()
        layers.append(DualLoRALinear(base, rank=rank, forget_rank=rank,
                                     alpha=16, dropout=0.0).double())
    return _perturb_down(nn.ModuleList(layers))


def _forward_tower(tower, x):
    for layer in tower:
        x = layer(x)
    return x


def _adapter_params(tower):
    retain, forget = [], []
    for layer in tower:
        retain += layer.get_retain_params()
        forget += layer.get_forget_params()
    return retain, forget


def _grouped_grads(tower):
    return [(p, None if p.grad is None else p.grad.detach().clone())
            for layer in tower for p in (layer.get_retain_params() + layer.get_forget_params())]


# --------------------------------------------------------------------------- #
# A per-sequence-normalized GRPO-like loss on the packed output
# --------------------------------------------------------------------------- #

def _seq_loss(y_packed, spans, adv, proj):
    """y_packed: (1, T, H). spans: list of (start, end, sample_id). adv: dict
    sample_id -> advantage. proj: (H,) fixed projection to a per-token scalar.

    Per sequence j: loss_j = adv_j * mean_t(s_t) over its span (per-sequence
    normalization, matching loss_type='grpo'). Returns mean_j(loss_j).
    """
    s = (y_packed[0] * proj).sum(-1)  # (T,)
    per_seq = []
    for (a, b, sid) in spans:
        per_seq.append(adv[sid] * s[a:b].mean())
    return torch.stack(per_seq).mean()


# --------------------------------------------------------------------------- #
# Stock homogeneous-microbatch reference (per class: fwd+bwd with param hooks)
# --------------------------------------------------------------------------- #

def _stock_reference(tower, x_full, spans_full, classes, adv, proj,
                     train_fs, exclusive, scale_denom):
    """Replicate _dynamic_microbatch_forward_backward's per-class passes."""
    retain_params, forget_params = _adapter_params(tower)
    for p in retain_params + forget_params:
        p.grad = None

    by_class = {"coherence": [], "good": [], "bad": []}
    for j, (a, b, sid) in enumerate(spans_full):
        by_class[classes[sid]].append(j)

    for cls, seq_idxs in by_class.items():
        if not seq_idxs:
            continue
        # Slice the packed tensor + spans for this class, re-based to [0, T_cls).
        chunks, sub_spans, off = [], [], 0
        for j in seq_idxs:
            a, b, sid = spans_full[j]
            chunks.append(x_full[:, a:b, :])
            sub_spans.append((off, off + (b - a), sid))
            off += (b - a)
        x_cls = torch.cat(chunks, dim=1)

        if cls == "coherence":
            set_scales(tower, retain_scale=1.0, forget_scale=0.0)
            hooks = [p.register_hook(lambda g: torch.zeros_like(g)) for p in forget_params]
        elif cls == "good":
            set_scales(tower, retain_scale=1.0, forget_scale=train_fs)
            hooks = ([p.register_hook(lambda g: torch.zeros_like(g)) for p in forget_params]
                     if exclusive else [])
        else:  # bad
            set_scales(tower, retain_scale=1.0, forget_scale=train_fs)
            hooks = [p.register_hook(lambda g: torch.zeros_like(g)) for p in retain_params]

        y = _forward_tower(tower, x_cls)
        loss = _seq_loss(y, sub_spans, adv, proj)
        scale = len(seq_idxs) / scale_denom
        (loss * scale).backward()
        for h in hooks:
            h.remove()

    set_scales(tower, retain_scale=1.0, forget_scale=1.0)
    return _grouped_grads(tower)


# --------------------------------------------------------------------------- #
# Fused single-pass (per-token forward forget-scale + activation-grad masks)
# --------------------------------------------------------------------------- #

def _fused(tower, x_full, spans_full, classes, adv, proj,
           train_fs, exclusive, scale_denom):
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
            ffs, rgm, fgm = train_fs, 1.0, (0.0 if exclusive else 1.0)
        else:
            ffs, rgm, fgm = train_fs, 0.0, 1.0
        forget_fwd[a:b] = ffs
        retain_gm[a:b] = rgm
        forget_gm[a:b] = fgm

    set_scales(tower, retain_scale=1.0, forget_scale=1.0)  # forget_scale ignored when fused
    n_kept = len(spans_full)
    set_fused_routing(forget_fwd.view(1, T, 1), retain_gm.view(1, T, 1), forget_gm.view(1, T, 1))
    try:
        y = _forward_tower(tower, x_full)
        loss = _seq_loss(y, spans_full, adv, proj)
        (loss * (n_kept / scale_denom)).backward()
    finally:
        clear_fused_routing()
    return _grouped_grads(tower)


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #

def _make_batch(hidden, seed):
    torch.manual_seed(seed)
    # 6 sequences of varying length: 2 coherence, 2 good, 2 bad.
    classes = {0: "coherence", 1: "coherence", 2: "good", 3: "good", 4: "bad", 5: "bad"}
    lengths = {0: 3, 1: 5, 2: 4, 3: 2, 4: 6, 5: 3}
    adv = {sid: float(torch.randn(1).item()) for sid in classes}
    spans, off = [], 0
    for sid in sorted(classes):  # pack order: coh, coh, good, good, bad, bad
        L = lengths[sid]
        spans.append((off, off + L, sid))
        off += L
    T = off
    x = torch.randn(1, T, hidden, dtype=torch.float64, requires_grad=False)
    proj = torch.randn(hidden, dtype=torch.float64)
    return x, spans, classes, adv, proj


def _compare(name, g_stock, g_fused):
    worst = 0.0
    for (p_s, gs), (p_f, gf) in zip(g_stock, g_fused):
        assert p_s is p_f
        if gs is None and gf is None:
            continue
        assert (gs is None) == (gf is None), f"{name}: grad presence differs"
        denom = gs.abs().max().item()
        diff = (gs - gf).abs().max().item()
        rel = diff / denom if denom > 0 else diff
        worst = max(worst, rel)
    print(f"  {name}: worst relative grad diff = {worst:.3e}")
    assert worst < 1e-9, f"{name}: fused != stock (worst rel {worst:.3e})"


def _run_case(tower_fn, name, exclusive):
    hidden = 8
    x, spans, classes, adv, proj = _make_batch(hidden, seed=7)
    train_fs = 0.7
    scale_denom = len(spans) + 2  # > n_kept: emulate dropped (e.g. unverified-coh) samples

    tower = tower_fn()
    g_stock = _stock_reference(tower, x, spans, classes, adv, proj,
                               train_fs, exclusive, scale_denom)
    g_fused = _fused(tower, x, spans, classes, adv, proj,
                     train_fs, exclusive, scale_denom)
    _compare(f"{name} (exclusive={exclusive})", g_stock, g_fused)


def test_mlp_classic():
    _run_case(lambda: _build_mlp_tower(3, 8, 4, seed=1), "MLP", exclusive=False)


def test_mlp_exclusive():
    _run_case(lambda: _build_mlp_tower(3, 8, 4, seed=1), "MLP", exclusive=True)


def test_lora_classic():
    _run_case(lambda: _build_lora_tower(3, 8, 2, seed=2), "LoRA", exclusive=False)


def test_lora_exclusive():
    _run_case(lambda: _build_lora_tower(3, 8, 2, seed=2), "LoRA", exclusive=True)


if __name__ == "__main__":
    print("Fused gradient-routing equivalence (stock per-class passes vs fused single pass):")
    test_mlp_classic()
    test_mlp_exclusive()
    test_lora_classic()
    test_lora_exclusive()
    print("ALL PASS ✓")
