"""Correctness test for the split-moment pre-routing gradient capture.

The split-moment feature (train.py --split_moment) needs, from ONE backward:
  - g_post  -> .grad                 the ROUTED gradient (balanced: retain bad=0,
                                      forget bad=2) — feeds Adam's first moment m.
  - g_pre   -> ._pre_routing_grad    the NATURAL gradient (every token scale 1,
                                      both adapters) — feeds Adam's second moment v.

PreRoutingGradAccumulator (gradient_routing.py) reconstructs g_pre from each
DualLoRALinear's saved input x and output-grad g during the decoupled backward.
This test pins two invariants on tiny CPU LoRA towers:
  (A) g_pre (captured) == the .grad of a NATURAL fused backward (all masks 1).
  (B) g_post (.grad with the capture installed) == g_post WITHOUT the capture
      (the capture's hooks must not perturb the training gradient).

float32 (not the equivalence test's float64) because the accumulator reconstructs
in float32, matching real runs; tolerance is set to float32 round-off.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import (  # noqa: E402
    DualLoRALinear, DualMLPAdapter, set_scales, set_fused_routing,
    clear_fused_routing, PreRoutingGradAccumulator,
)


def _build_lora_tower(n_layers, hidden, rank, seed):
    torch.manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        base = nn.Linear(hidden, hidden, bias=False)
        layers.append(DualLoRALinear(base, rank=rank, forget_rank=rank,
                                     alpha=16, dropout=0.0))
    # lora_B zero-inits, so perturb it for non-trivial grads on every param.
    with torch.no_grad():
        for l in layers:
            l.lora_B_retain.normal_(0, 0.1)
            l.lora_B_forget.normal_(0, 0.1)
    return nn.ModuleList(layers)


def _build_mlp_tower(n_layers, hidden, neurons, seed):
    torch.manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        base = nn.Linear(hidden, hidden, bias=False)
        layers.append(DualMLPAdapter(base, hidden, neurons, neurons))
    # down_{role} zero-inits, so perturb for non-trivial grads on every param.
    with torch.no_grad():
        for l in layers:
            l.down_retain.weight.normal_(0, 0.1)
            l.down_forget.weight.normal_(0, 0.1)
    return nn.ModuleList(layers)


def _forward(tower, x):
    for l in tower:
        x = l(x)
    return x


def _seq_loss(y, spans, adv, proj):
    s = (y[0] * proj).sum(-1)  # (T,)
    return torch.stack([adv[sid] * s[a:b].mean() for (a, b, sid) in spans]).mean()


def _make_batch(hidden, seed, with_coherence=False):
    torch.manual_seed(seed)
    if with_coherence:
        classes = {0: "coherence", 1: "good", 2: "good", 3: "bad", 4: "bad", 5: "coherence"}
        lengths = {0: 3, 1: 4, 2: 2, 3: 5, 4: 3, 5: 4}
    else:
        classes = {0: "good", 1: "good", 2: "bad", 3: "bad"}
        lengths = {0: 4, 1: 2, 2: 5, 3: 3}
    adv = {sid: float(torch.randn(1).item()) for sid in classes}
    spans, off = [], 0
    for sid in sorted(classes):
        spans.append((off, off + lengths[sid], sid))
        off += lengths[sid]
    x = torch.randn(1, off, hidden)
    proj = torch.randn(hidden)
    return x, spans, classes, adv, proj


def _routing_tensors(spans, T, classes, train_fs, retain_bad, forget_bad):
    forget_fwd = torch.zeros(T)
    retain_gm = torch.zeros(T)
    forget_gm = torch.zeros(T)
    for (a, b, sid) in spans:
        if classes[sid] == "coherence":   # retain-only: forget forward off
            ffs, rgm, fgm = 0.0, 1.0, 0.0
        elif classes[sid] == "good":
            ffs, rgm, fgm = train_fs, 1.0, 1.0
        else:  # bad
            ffs, rgm, fgm = train_fs, retain_bad, forget_bad
        forget_fwd[a:b] = ffs
        retain_gm[a:b] = rgm
        forget_gm[a:b] = fgm
    return forget_fwd, retain_gm, forget_gm


def _params(tower):
    return [p for l in tower for p in (l.get_retain_params() + l.get_forget_params())]


def _fused_backward(tower, x, spans, classes, adv, proj, train_fs,
                    retain_bad, forget_bad, accumulator=None):
    for p in _params(tower):
        p.grad = None
    if accumulator is not None:
        accumulator.reset()
    T = x.shape[1]
    ff, rg, fg = _routing_tensors(spans, T, classes, train_fs, retain_bad, forget_bad)
    set_scales(tower, retain_scale=1.0, forget_scale=1.0)  # forget_scale ignored when fused
    set_fused_routing(ff.view(1, T, 1), rg.view(1, T, 1), fg.view(1, T, 1))
    try:
        y = _forward(tower, x)
        loss = _seq_loss(y, spans, adv, proj)
        loss.backward()
        if accumulator is not None:
            accumulator.flush(ff.view(1, T, 1))
    finally:
        clear_fused_routing()
    return {id(p): (None if p.grad is None else p.grad.detach().clone())
            for p in _params(tower)}


def _compare(name, ref, got):
    worst = 0.0
    for k, r in ref.items():
        g = got[k]
        assert (r is None) == (g is None), f"{name}: grad presence differs"
        if r is None:
            continue
        denom = r.abs().max().item()
        diff = (r - g).abs().max().item()
        worst = max(worst, diff / denom if denom > 0 else diff)
    print(f"  {name}: worst relative diff = {worst:.3e}")
    assert worst < 1e-4, f"{name}: mismatch (worst rel {worst:.3e})"


def _run_capture_case(name, tower, hidden, train_fs, with_coherence=False):
    x, spans, classes, adv, proj = _make_batch(hidden, seed=7, with_coherence=with_coherence)

    # Run A: BALANCED routing (retain bad=0, forget bad=2) WITH the capture.
    # Coherence samples (if any) are retain-only: forget forward off, retain mask 1.
    cap = PreRoutingGradAccumulator(tower)
    g_post = _fused_backward(tower, x, spans, classes, adv, proj, train_fs,
                             retain_bad=0.0, forget_bad=2.0, accumulator=cap)
    g_pre = {id(p): getattr(p, "_pre_routing_grad", None) for p in _params(tower)}
    cap.remove()

    # (A) g_pre == NATURAL backward (all GATE masks 1, both adapters). The forward
    # forget-scale stays per-class (0 on coherence), so coherence contributes
    # weight-1 to retain and 0 to forget in g_pre.
    g_pre_ref = _fused_backward(tower, x, spans, classes, adv, proj, train_fs,
                                retain_bad=1.0, forget_bad=1.0, accumulator=None)
    _compare(f"{name}: g_pre vs natural backward", g_pre_ref, g_pre)

    # (B) g_post (with capture) == g_post (without capture) — capture is observational.
    g_post_ref = _fused_backward(tower, x, spans, classes, adv, proj, train_fs,
                                 retain_bad=0.0, forget_bad=2.0, accumulator=None)
    _compare(f"{name}: g_post capture-vs-clean", g_post_ref, g_post)

    # Sanity: g_pre and g_post genuinely differ (else the test is vacuous).
    any_diff = any(
        g_pre[id(p)] is not None and g_post[id(p)] is not None
        and (g_pre[id(p)] - g_post[id(p)]).abs().max().item() > 1e-6
        for p in _params(tower))
    assert any_diff, f"{name}: g_pre and g_post identical — routing had no effect"


def test_split_moment_capture_lora():
    _run_capture_case("LoRA", _build_lora_tower(3, 8, 2, seed=2), hidden=8, train_fs=0.7)


def test_split_moment_capture_lora_coherence():
    _run_capture_case("LoRA+coh", _build_lora_tower(3, 8, 2, seed=2), hidden=8,
                      train_fs=0.7, with_coherence=True)


def test_split_moment_capture_mlp_coherence():
    _run_capture_case("MLP+coh", _build_mlp_tower(3, 8, 4, seed=2), hidden=8,
                      train_fs=0.7, with_coherence=True)


def test_split_moment_capture_mlp():
    _run_capture_case("MLP", _build_mlp_tower(3, 8, 4, seed=2), hidden=8, train_fs=0.7)


if __name__ == "__main__":
    test_split_moment_capture_lora()
    test_split_moment_capture_mlp()
    test_split_moment_capture_lora_coherence()
    test_split_moment_capture_mlp_coherence()
    print("PASS ✓")
