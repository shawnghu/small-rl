"""
2-backward v=a_v isolation proof for the graft-port slow path (routing_lambda != 1).

Imports the REAL gradient_routing.DualMLPAdapter + PreRoutingGradAccumulator +
set_fused_routing/clear_fused_routing for faithfulness. Builds a tiny base+dual-MLP
adapter, a hand GRPO loss (PG with PPO clip + beta*k3 KL), and TWO per-token
advantages:
   a_m  (zero-mean-retain weighted baseline) -> first moment m  (DECOUPLE-MASKED grad)
   a_v  (lambda-independent nonflagged baseline) -> second moment v (NATURAL grad)
At lambda!=1 a_m != a_v, so the orchestration needs a SECOND honest backward at a_v
to feed v (proven necessary by /tmp/v_feasibility.py).

PROVES, by comparison to an independent reference:
   m == masked-grad(a_m)        (the routed/decoupled param-grad at a_m, in .grad)
   v == natural-grad(a_v)       (the unmasked adapter param-grad at a_v, in _pre_routing_grad)
with NO cross-contamination between the two backwards, across
   (on-policy / off-policy) x (beta in {0, 0.05}).

Two production-faithful orchestrations are validated:
   Variant A: two forwards (fused-for-m, natural-for-v) -- uses the accumulator EXACTLY
              as production does today (single forward+backward per capture).
   Variant B: ONE shared fused forward, TWO backwards (m at a_m retain_graph=True;
              save+zero .grad; rearm accumulator [drop g_m, keep x]; v-backward at a_v;
              flush; restore .grad=m). This is the proposed _fused_forward_backward edit.
"""
import sys, copy
sys.path.insert(0, "/workspace/small-rl")
import torch
import torch.nn as nn
from gradient_routing import (DualMLPAdapter, PreRoutingGradAccumulator,
                              set_fused_routing, clear_fused_routing)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# ---- dims ----
S, T, D, V = 3, 5, 6, 9          # seqs, tokens, hidden, vocab
R_N, F_N = 3, 2                  # retain / forget adapter neurons (unequal -> kappa != 2,2)

# ---- frozen "base mlp" the adapter wraps (identity-ish linear) ----
base_mlp = nn.Linear(D, D, bias=False)
nn.init.eye_(base_mlp.weight)

adapter = DualMLPAdapter(base_mlp, hidden_size=D, retain_neurons=R_N, forget_neurons=F_N)
# down_* init to zeros => adapter output 0 and gate/up grads vanish. Randomize so EVERY
# adapter param carries a nonzero gradient (a meaningful isolation test).
for lin in (adapter.gate_retain, adapter.up_retain, adapter.down_retain,
            adapter.gate_forget, adapter.up_forget, adapter.down_forget):
    nn.init.normal_(lin.weight, std=0.4)

# ---- cross-token-mixing upper net: makes dL/dy non-diagonal across tokens, so a
#      per-token decouple mask genuinely differs from any global scale ----
Mmix = torch.randn(T, T)
Wh   = torch.randn(D, V)
chosen   = torch.randint(0, V, (S, T))
ref_logp = torch.log_softmax(torch.randn(S, T, V), -1).gather(
    -1, chosen.unsqueeze(-1)).squeeze(-1)

def upper(module_out):                      # module_out: (S,T,D) -> logp (S,T)
    h = torch.tanh(module_out)
    hmix = torch.einsum('tu,sud->std', Mmix, h)
    logits = hmix @ Wh
    return torch.log_softmax(logits, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)

X = torch.randn(S, T, D)

# ---- per-token routing tensors (1,T,1): a classic lambda<1 redistribution.
#      All T tokens here belong to "bad/detected" routing samples, so retain is
#      down-weighted (1-lam) and forget up-weighted (1+lam(kF-1)). forget fwd-on. ----
from advantages import adapter_kappas, routing_grad_mask_weights
kR, kF = adapter_kappas(R_N, F_N)
lam = 0.5
_, _, rgm_bad, fgm_bad = routing_grad_mask_weights("classic", lam, kR, kF)
forget_fwd = torch.full((1, T, 1), 1.0)          # train_forget_scale (forget fwd-on)
retain_gm  = torch.full((1, T, 1), float(rgm_bad))
forget_gm  = torch.full((1, T, 1), float(fgm_bad))
print(f"kappa=({kR:.3f},{kF:.3f})  lam={lam}  masks: retain_gm={rgm_bad:.3f} forget_gm={fgm_bad:.3f}\n")

# ---- two per-sequence advantages; seq signs straddle so a_m != a_v honestly ----
a_m = torch.tensor([1.0, -0.5,  0.8])     # weighted-baseline (m)
a_v = torch.tensor([0.6,  0.3, -0.4])     # nonflagged-baseline (v); lambda-independent

params = adapter.get_retain_params() + adapter.get_forget_params()

def grpo_loss(logp, A, beta, shift, eps=0.2):
    old = logp.detach() - shift                  # off-policy: ratio0 = exp(shift)
    ratio = torch.exp(logp - old)
    Ab = A.view(S, 1)
    pg = -torch.minimum(ratio * Ab, torch.clamp(ratio, 1 - eps, 1 + eps) * Ab)
    k3 = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
    return (pg + beta * k3).sum()

def relerr(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-300)

# =====================================================================================
# INDEPENDENT REFERENCE (no accumulator, no set_fused_routing): defines truth.
#   m_ref = decouple-masked param-grad at a_m  (mask applied to the dL/dy seed)
#   v_ref = natural (unmasked) param-grad at a_v
# =====================================================================================
def reference(beta, shift):
    # natural module output (fused OFF) -> exactly base + retain_core + forget_core*ffs
    clear_fused_routing()
    nat = adapter.natural_adapter_output(X, forget_fwd)        # adapter contribution only
    module_out = base_mlp(X) + nat
    logp = upper(module_out)

    # m: seed = dL_{a_m}/d(module_out), masked per adapter, backprop'd through natural adapter
    Lm = grpo_loss(logp, a_m, beta, shift)
    g_m = torch.autograd.grad(Lm, module_out, retain_graph=True)[0]            # (S,T,D)
    rmask = retain_gm.permute(0, 2, 1).reshape(1, T, 1)                        # (1,T,1)
    fmask = forget_gm.permute(0, 2, 1).reshape(1, T, 1)
    m_ret = torch.autograd.grad(nat, adapter.get_retain_params(),
                                grad_outputs=(rmask * g_m), retain_graph=True)
    m_for = torch.autograd.grad(nat, adapter.get_forget_params(),
                                grad_outputs=(fmask * g_m), retain_graph=True)
    m_ref = list(m_ret) + list(m_for)

    # v: seed = dL_{a_v}/d(module_out), NO mask, natural adapter
    Lv = grpo_loss(logp, a_v, beta, shift)
    g_v = torch.autograd.grad(Lv, module_out, retain_graph=True)[0]
    v_ref = list(torch.autograd.grad(nat, params, grad_outputs=g_v))
    return m_ref, v_ref

# =====================================================================================
# VARIANT A -- two forwards (accumulator used EXACTLY as production does today).
# =====================================================================================
def variant_A(beta, shift):
    # --- m: fused forward + backward at a_m, read .grad ---
    for p in params: p.grad = None
    set_fused_routing(forget_fwd, retain_gm, forget_gm)
    out = adapter(X)
    logp = upper(out)
    Lm = grpo_loss(logp, a_m, beta, shift)
    Lm.backward()
    clear_fused_routing()
    m = [p.grad.detach().clone() for p in params]

    # --- v: separate NATURAL forward + backward at a_v with the accumulator ---
    cap = PreRoutingGradAccumulator(adapter)
    cap.reset()
    set_fused_routing(forget_fwd, retain_gm, forget_gm)   # forward goes through fused branch
    out2 = adapter(X)                                     # pre_hook saves x
    logp2 = upper(out2)
    Lv = grpo_loss(logp2, a_v, beta, shift)
    for p in params: p.grad = None
    Lv.backward()                                         # _bwd captures g_v
    cap.flush(forget_fwd)                                 # -> _pre_routing_grad = natural v at a_v
    clear_fused_routing()
    v = [getattr(p, "_pre_routing_grad").detach().clone() for p in params]
    cap.remove()
    return m, v

# =====================================================================================
# VARIANT B -- ONE shared fused forward, TWO backwards (the proposed train.py edit).
#   bw1 at a_m, retain_graph=True -> .grad=m ; rearm accumulator (drop g_m, KEEP x);
#   save+zero .grad ; bw2 at a_v -> capture g_v -> flush -> v ; restore .grad=m.
# rearm() is the one small PreRoutingGradAccumulator addition the slow path needs.
# =====================================================================================
def _rearm(cap):
    """Now a real method on PreRoutingGradAccumulator — exercise it directly so
    the proto regression-tests the shipped implementation."""
    cap.rearm()

def variant_B(beta, shift):
    cap = PreRoutingGradAccumulator(adapter)
    cap.reset()
    for p in params: p.grad = None
    set_fused_routing(forget_fwd, retain_gm, forget_gm)
    out = adapter(X)                       # ONE forward (pre_hook saves x)
    logp = upper(out)

    # bw1 at a_m (masked) -> .grad = m ; captures g_m (to be discarded)
    Lm = grpo_loss(logp, a_m, beta, shift)
    Lm.backward(retain_graph=True)
    m = [p.grad.detach().clone() for p in params]     # save m
    for p in params: p.grad = None                    # zero so bw2 garbage doesn't pollute m
    _rearm(cap)                                        # drop g_m captures, keep x

    # bw2 at a_v (natural via accumulator) -> v
    Lv = grpo_loss(logp, a_v, beta, shift)            # same graph, new advantage
    Lv.backward()                                      # captures g_v
    cap.flush(forget_fwd)
    v = [getattr(p, "_pre_routing_grad").detach().clone() for p in params]

    for p, g in zip(params, m): p.grad = g            # restore .grad = m (optimizer reads this)
    clear_fused_routing()
    cap.remove()
    return m, v

# =====================================================================================
print(f"{'regime':20} {'m relerr (A)':>14} {'v relerr (A)':>14} "
      f"{'m relerr (B)':>14} {'v relerr (B)':>14} {'A==B v':>10}")
worst = 0.0
for beta in (0.0, 0.05):
    for shift, tag in ((0.0, "on-policy"), (0.4, "off-policy")):
        m_ref, v_ref = reference(beta, shift)
        mA, vA = variant_A(beta, shift)
        mB, vB = variant_B(beta, shift)
        em_A = max(relerr(a, b) for a, b in zip(mA, m_ref))
        ev_A = max(relerr(a, b) for a, b in zip(vA, v_ref))
        em_B = max(relerr(a, b) for a, b in zip(mB, m_ref))
        ev_B = max(relerr(a, b) for a, b in zip(vB, v_ref))
        eAB  = max(relerr(a, b) for a, b in zip(vA, vB))
        worst = max(worst, em_A, ev_A, em_B, ev_B, eAB)
        print(f"beta={beta:<4} {tag:11} {em_A:14.2e} {ev_A:14.2e} "
              f"{em_B:14.2e} {ev_B:14.2e} {eAB:10.2e}")

# Cross-contamination guards (beta=0, on-policy: PG dominates so the a_m vs a_v
# difference is fully visible): v must ride a_v, not a_m; and m's decouple mask must bite.
print()
m_ref, v_ref = reference(0.0, 0.0)
clear_fused_routing()
nat = adapter.natural_adapter_output(X, forget_fwd); mo = base_mlp(X) + nat
g_am = torch.autograd.grad(grpo_loss(upper(mo), a_m, 0.0, 0.0), mo, retain_graph=True)[0]
v_if_rode_am = list(torch.autograd.grad(nat, params, grad_outputs=g_am))   # natural grad AT a_m
print(f"sanity: v(a_v) vs natural-grad(a_m)  relerr = "
      f"{max(relerr(a,b) for a,b in zip(v_ref, v_if_rode_am)):.3e}  (LARGE -> v truly rides a_v, not a_m)")
print(f"sanity: m(a_m) vs natural-grad(a_m)  relerr = "
      f"{max(relerr(a,b) for a,b in zip(m_ref, v_if_rode_am)):.3e}  (LARGE -> decouple mask truly bit)")
print(f"\nWORST relerr across all cells = {worst:.2e}  ->  {'PASS' if worst < 1e-10 else 'FAIL'}")
