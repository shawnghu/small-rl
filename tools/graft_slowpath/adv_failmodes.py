"""
Probe the three load-bearing MECHANICAL invariants of the slow path. For each,
DELIBERATELY break it and characterize the failure: loud (throws) or SILENT (wrong
numbers, no error). Silent failures are the real danger for experimental correctness.

  (A) retain_graph=True on bw1 omitted -> bw2's second liger call on the freed graph.
  (B) rearm() omitted -> bw2's _bwd sees x=None (popped by bw1) and skips v capture.
  (C) snapshot/restore of .grad omitted -> bw2's masked-grad(a_v) contaminates m.
"""
import sys
sys.path.insert(0, "/workspace/small-rl")
import torch, torch.nn as nn, io, contextlib
from gradient_routing import (DualMLPAdapter, PreRoutingGradAccumulator,
                              set_fused_routing, clear_fused_routing)
from advantages import adapter_kappas, routing_grad_mask_weights

torch.set_default_dtype(torch.float64); torch.manual_seed(2)
S, T, D, Vv = 3, 5, 6, 9
R_N, F_N = 3, 2
kR, kF = adapter_kappas(R_N, F_N); lam = 0.5
base = nn.Linear(D, D, bias=False); nn.init.eye_(base.weight)
ad = DualMLPAdapter(base, hidden_size=D, retain_neurons=R_N, forget_neurons=F_N)
for lin in (ad.gate_retain, ad.up_retain, ad.down_retain, ad.gate_forget, ad.up_forget, ad.down_forget):
    nn.init.normal_(lin.weight, std=0.4)
params = ad.get_retain_params() + ad.get_forget_params()
Mmix = torch.randn(T, T); Wh = torch.randn(D, Vv)
X = torch.randn(S, T, D); chosen = torch.randint(0, Vv, (S, T))
ref_logp = torch.log_softmax(torch.randn(S, T, Vv), -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
_, _, rgm, fgm = routing_grad_mask_weights("classic", lam, kR, kF)
forget_fwd = torch.full((1, T, 1), 0.7)
retain_gm = torch.full((1, T, 1), float(rgm)); forget_gm = torch.full((1, T, 1), float(fgm))
a_m = torch.tensor([1.0, -0.5, 0.8]); a_v = torch.tensor([0.6, 0.3, -0.4])

def upper(out):
    h = torch.tanh(out); hmix = torch.einsum('tu,sud->std', Mmix, h)
    return torch.log_softmax(hmix @ Wh, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
def loss(logp, A, beta=0.05, shift=0.4, eps=0.2):
    old = logp.detach() - shift; r = torch.exp(logp - old); Ab = A.view(S, 1)
    pg = -torch.minimum(r * Ab, torch.clamp(r, 1 - eps, 1 + eps) * Ab)
    k3 = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
    return (pg + beta * k3).sum()
def relerr(a, b): return (a - b).norm().item() / (b.norm().item() + 1e-300)
def _rearm(cap):
    for mod, x, g in cap._captures: cap._saved[id(mod)] = x
    cap._captures = []

# reference v = natural-grad(a_v)
clear_fused_routing()
nat = ad.natural_adapter_output(X, forget_fwd); out = base(X) + nat
g_v = torch.autograd.grad(loss(upper(out), a_v), out, retain_graph=True)[0]
v_ref = list(torch.autograd.grad(nat, params, grad_outputs=g_v, retain_graph=True))
g_m = torch.autograd.grad(loss(upper(out), a_m), out, retain_graph=True)[0]
m_ret = torch.autograd.grad(nat, ad.get_retain_params(), grad_outputs=retain_gm * g_m, retain_graph=True)
m_for = torch.autograd.grad(nat, ad.get_forget_params(), grad_outputs=forget_gm * g_m, retain_graph=True)
m_ref = list(m_ret) + list(m_for)

# ---------- (A) omit retain_graph on bw1 ----------
print("(A) bw1 WITHOUT retain_graph=True:")
cap = PreRoutingGradAccumulator(ad); cap.reset()
for p in params: p.grad = None
set_fused_routing(forget_fwd, retain_gm, forget_gm)
o = ad(X); lp = upper(o)
with contextlib.redirect_stderr(io.StringIO()):
    loss(lp, a_m).backward()  # no retain_graph
    _rearm(cap)
    try:
        loss(lp, a_v).backward()
        print("    bw2 SUCCEEDED (unexpected)")
    except RuntimeError as e:
        print(f"    LOUD RuntimeError: {str(e).splitlines()[0][:70]}")
clear_fused_routing(); cap.remove()

# ---------- (B) omit rearm() ----------
print("\n(B) rearm() OMITTED between bw1 and bw2:")
cap = PreRoutingGradAccumulator(ad); cap.reset()
for p in params: p.grad = None
set_fused_routing(forget_fwd, retain_gm, forget_gm)
o = ad(X); lp = upper(o)
with contextlib.redirect_stderr(io.StringIO()):
    loss(lp, a_m).backward(retain_graph=True)
    # NO rearm -> _saved already popped by bw1's _bwd; _captures still holds g_m
    n_before = len(cap._captures)
    loss(lp, a_v).backward()  # _bwd sees x=None, returns early (silent)
    cap.flush(forget_fwd)
v_norearm = [getattr(p, "_pre_routing_grad", None) for p in params]
clear_fused_routing(); cap.remove()
if any(x is None for x in v_norearm):
    print("    v has None entries (capture skipped)")
ev = max(relerr(a if a is not None else torch.zeros_like(b), b) for a, b in zip(v_norearm, v_ref))
print(f"    captures held after bw1 (no rearm) = {n_before} (g_m, NOT g_v)")
print(f"    v relerr vs natural(a_v) = {ev:.3e}  <- SILENT WRONG (flush used g_m+stale x or skipped)")

# ---------- (C) omit snapshot/restore of .grad ----------
print("\n(C) snapshot/restore of .grad OMITTED:")
cap = PreRoutingGradAccumulator(ad); cap.reset()
for p in params: p.grad = None
set_fused_routing(forget_fwd, retain_gm, forget_gm)
o = ad(X); lp = upper(o)
with contextlib.redirect_stderr(io.StringIO()):
    loss(lp, a_m).backward(retain_graph=True)
    _rearm(cap)
    loss(lp, a_v).backward()  # accumulates masked-grad(a_v) ONTO .grad
    cap.flush(forget_fwd)
m_contam = [p.grad.detach().clone() for p in params]
clear_fused_routing(); cap.remove()
em = max(relerr(a, b) for a, b in zip(m_contam, m_ref))
print(f"    m relerr vs masked(a_m) = {em:.3e}  <- SILENT WRONG (m = masked(a_m)+masked(a_v))")

# ---------- correct version for contrast ----------
print("\n(OK) full correct orchestration:")
cap = PreRoutingGradAccumulator(ad); cap.reset()
for p in params: p.grad = None
set_fused_routing(forget_fwd, retain_gm, forget_gm)
o = ad(X); lp = upper(o)
with contextlib.redirect_stderr(io.StringIO()):
    loss(lp, a_m).backward(retain_graph=True)
    snap = {p: p.grad.detach().clone() for p in params}
    _rearm(cap)
    loss(lp, a_v).backward()
    cap.flush(forget_fwd)
    for p in params: p.grad = snap[p]
m = [p.grad.detach().clone() for p in params]
v = [getattr(p, "_pre_routing_grad").detach().clone() for p in params]
clear_fused_routing(); cap.remove()
print(f"    m relerr = {max(relerr(a,b) for a,b in zip(m,m_ref)):.2e}   v relerr = {max(relerr(a,b) for a,b in zip(v,v_ref)):.2e}")
