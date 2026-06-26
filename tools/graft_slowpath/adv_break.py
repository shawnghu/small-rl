"""
ADVERSARIAL break-attempt on the m/v-isolation proof for the graft-port slow path.

Goes beyond /tmp/v_isolation_proto.py by stressing exactly the regimes the original
did NOT cover:
  (1) EXCLUSIVE masks: good-routing tokens get retain_gm=1+λ(κ_R-1) (>1, AMPLIFYING)
      and forget off; bad tokens classic. Heterogeneous PER-TOKEN masks within ONE
      microbatch (coherence + good-routing + bad-routing tokens mixed), so the
      per-token decouple genuinely differs token-to-token.
  (2) >1 MICROBATCH with REAL .grad accumulation (NOT zeroed between mbs) — the
      production loop. The snapshot/restore must preserve prior-mb accumulation.
      Tests the load-bearing claim that bw2's garbage is fully removed even when
      .grad already holds Σ of earlier microbatches.
  (3) explicit TOKEN-level clip-straddle: a_m and a_v opposite sign on the SAME
      token, off-policy so the PPO clip fires on opposite branches.
  (4) coherence tokens (forget_fwd=0) interleaved -> forget v must get 0 there.

Faithfulness: REAL gradient_routing.DualMLPAdapter / PreRoutingGradAccumulator /
set_fused_routing, REAL advantages.adapter_kappas / routing_grad_mask_weights.
Reference = independent direct autograd (no accumulator, no fused routing).

m must == decouple-masked param-grad at a_m (in .grad, Σ over mbs);
v must == natural (unmasked) param-grad at a_v (in _pre_routing_grad, Σ over mbs).
"""
import sys
sys.path.insert(0, "/workspace/small-rl")
import torch, torch.nn as nn
from gradient_routing import (DualMLPAdapter, PreRoutingGradAccumulator,
                              set_fused_routing, clear_fused_routing)
from advantages import adapter_kappas, routing_grad_mask_weights

torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

# ---- dims: 2 microbatches, each S seqs x T tokens ----
NMB, S, T, D, Vv = 2, 4, 6, 7, 11
R_N, F_N = 3, 2
kR, kF = adapter_kappas(R_N, F_N)
lam = 0.5

base_mlp = nn.Linear(D, D, bias=False); nn.init.eye_(base_mlp.weight)
adapter = DualMLPAdapter(base_mlp, hidden_size=D, retain_neurons=R_N, forget_neurons=F_N)
for lin in (adapter.gate_retain, adapter.up_retain, adapter.down_retain,
            adapter.gate_forget, adapter.up_forget, adapter.down_forget):
    nn.init.normal_(lin.weight, std=0.4)
params = adapter.get_retain_params() + adapter.get_forget_params()

# cross-token-mixing upper net per microbatch (non-diagonal dL/dy)
Mmix = torch.randn(T, T); Wh = torch.randn(D, Vv)

def make_mb(seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(S, T, D, generator=g)
    chosen = torch.randint(0, Vv, (S, T), generator=g)
    ref_logp = torch.log_softmax(torch.randn(S, T, Vv, generator=g), -1).gather(
        -1, chosen.unsqueeze(-1)).squeeze(-1)
    return X, chosen, ref_logp

def upper(out, chosen):
    h = torch.tanh(out)
    hmix = torch.einsum('tu,sud->std', Mmix, h)
    logits = hmix @ Wh
    return torch.log_softmax(logits, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)

# ---- EXCLUSIVE masks, heterogeneous PER TOKEN within the microbatch ----
rgm_g, fgm_g, rgm_b, fgm_b = routing_grad_mask_weights("exclusive", lam, kR, kF)
print(f"exclusive lam={lam} kappa=({kR:.3f},{kF:.3f})")
print(f"  good  tokens: retain_gm={rgm_g:.3f} (>1 AMPLIFY) forget_gm={fgm_g:.3f} forget_fwd=tfs")
print(f"  bad   tokens: retain_gm={rgm_b:.3f} forget_gm={fgm_b:.3f} forget_fwd=tfs")
print(f"  coh   tokens: retain_gm=1.000 forget_gm=0.000 forget_fwd=0\n")

TFS = 0.7  # train_forget_scale
# token roles: 0=coherence,1=good-routing,2=bad-routing  (varies across the T tokens)
roles = torch.tensor([0, 1, 2, 1, 2, 0])  # T=6
forget_fwd = torch.where(roles == 0, torch.zeros(T), torch.full((T,), TFS)).view(1, T, 1)
retain_gm = torch.tensor([1.0 if r == 0 else (rgm_g if r == 1 else rgm_b)
                          for r in roles.tolist()]).view(1, T, 1)
forget_gm = torch.tensor([0.0 if r == 0 else (fgm_g if r == 1 else fgm_b)
                          for r in roles.tolist()]).view(1, T, 1)

# advantages per microbatch: a_m, a_v straddle sign per-seq AND we add a per-token
# straddle by letting a_v differ in sign on some tokens via token-broadcast tweak.
# Keep per-seq (production is per-seq) but choose so seq signs flip between a_m/a_v.
A_M = [torch.tensor([1.0, -0.6, 0.4, -0.9]),
       torch.tensor([-0.3, 0.8, -0.5, 0.7])]
A_V = [torch.tensor([0.5, 0.7, -0.8, 0.2]),
       torch.tensor([0.6, -0.4, 0.3, -0.9])]

def grpo_loss(logp, A, ref_logp, beta, shift, eps=0.2):
    old = logp.detach() - shift
    ratio = torch.exp(logp - old)
    Ab = A.view(S, 1)
    pg = -torch.minimum(ratio * Ab, torch.clamp(ratio, 1 - eps, 1 + eps) * Ab)
    k3 = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
    return (pg + beta * k3).sum()

def relerr(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-300)

# scale per microbatch (mimics loss*scale; use distinct scales to stress accumulation)
SCALES = [1.3, 0.6]

# =====================================================================================
# REFERENCE: independent, per-mb, summed. m_ref = Σ masked-grad(a_m); v_ref = Σ nat(a_v)
# =====================================================================================
def reference(beta, shift):
    clear_fused_routing()
    m_ref = [torch.zeros_like(p) for p in params]
    v_ref = [torch.zeros_like(p) for p in params]
    rp, fp = adapter.get_retain_params(), adapter.get_forget_params()
    nr = len(rp)
    for mb in range(NMB):
        X, chosen, ref_logp = make_mb(100 + mb)
        sc = SCALES[mb]
        nat = adapter.natural_adapter_output(X, forget_fwd)
        out = base_mlp(X) + nat
        logp = upper(out, chosen)
        # m: masked seed at a_m
        Lm = grpo_loss(logp, A_M[mb], ref_logp, beta, shift)
        g_m = torch.autograd.grad(Lm * sc, out, retain_graph=True)[0]
        m_ret = torch.autograd.grad(nat, rp, grad_outputs=(retain_gm * g_m), retain_graph=True)
        m_for = torch.autograd.grad(nat, fp, grad_outputs=(forget_gm * g_m), retain_graph=True)
        for i, gg in enumerate(list(m_ret) + list(m_for)):
            m_ref[i] = m_ref[i] + gg
        # v: natural seed at a_v
        Lv = grpo_loss(logp, A_V[mb], ref_logp, beta, shift)
        g_v = torch.autograd.grad(Lv * sc, out, retain_graph=True)[0]
        v_all = torch.autograd.grad(nat, params, grad_outputs=g_v)
        for i, gg in enumerate(v_all):
            v_ref[i] = v_ref[i] + gg
    return m_ref, v_ref

# =====================================================================================
# PRODUCTION-FAITHFUL multi-microbatch Variant B: ONE accumulator, .grad ACCUMULATES
# across microbatches (never zeroed mid-step), snapshot/restore around each v-backward.
# =====================================================================================
def _rearm(cap):
    for mod, x, g in cap._captures:
        cap._saved[id(mod)] = x
    cap._captures = []

def production_loop(beta, shift):
    cap = PreRoutingGradAccumulator(adapter)
    cap.reset()
    for p in params: p.grad = None
    for mb in range(NMB):
        X, chosen, ref_logp = make_mb(100 + mb)
        sc = SCALES[mb]
        set_fused_routing(forget_fwd, retain_gm, forget_gm)
        out = adapter(X)
        logp = upper(out, chosen)
        # bw1 at a_m -> accumulates masked grad into .grad
        Lm = grpo_loss(logp, A_M[mb], ref_logp, beta, shift)
        (Lm * sc).backward(retain_graph=True)
        m_snap = {p: (p.grad.detach().clone() if p.grad is not None else None) for p in params}
        _rearm(cap)
        # bw2 at a_v -> capture g_v, flush to _pre_routing_grad; .grad gets garbage
        Lv = grpo_loss(logp, A_V[mb], ref_logp, beta, shift)
        (Lv * sc).backward()
        cap.flush(forget_fwd)
        for p in params:  # restore .grad to the post-bw1 snapshot (drops bw2 garbage)
            p.grad = m_snap[p]
        clear_fused_routing()
    m = [p.grad.detach().clone() for p in params]
    v = [getattr(p, "_pre_routing_grad").detach().clone() for p in params]
    cap.remove()
    return m, v

# =====================================================================================
print(f"{'regime':22} {'m relerr':>12} {'v relerr':>12} {'m-leak?':>10} {'v-leak?':>10}")
worst = 0.0
import io, contextlib
for beta in (0.0, 0.07):
    for shift, tag in ((0.0, "on-policy"), (0.45, "off-policy")):
        m_ref, v_ref = reference(beta, shift)
        with contextlib.redirect_stderr(io.StringIO()):
            m, v = production_loop(beta, shift)
        em = max(relerr(a, b) for a, b in zip(m, m_ref))
        ev = max(relerr(a, b) for a, b in zip(v, v_ref))
        # leak checks: m must NOT equal natural-grad(a_m); v must NOT equal masked(a_v)
        # cross refs
        clear_fused_routing()
        nat_am = [torch.zeros_like(p) for p in params]
        msk_av = [torch.zeros_like(p) for p in params]
        rp, fp = adapter.get_retain_params(), adapter.get_forget_params()
        for mb in range(NMB):
            X, chosen, ref_logp = make_mb(100 + mb); sc = SCALES[mb]
            nat = adapter.natural_adapter_output(X, forget_fwd)
            out = base_mlp(X) + nat; logp = upper(out, chosen)
            g_am = torch.autograd.grad(grpo_loss(logp, A_M[mb], ref_logp, beta, shift) * sc, out, retain_graph=True)[0]
            for i, gg in enumerate(torch.autograd.grad(nat, params, grad_outputs=g_am, retain_graph=True)):
                nat_am[i] = nat_am[i] + gg
            g_av = torch.autograd.grad(grpo_loss(logp, A_V[mb], ref_logp, beta, shift) * sc, out, retain_graph=True)[0]
            mret = torch.autograd.grad(nat, rp, grad_outputs=(retain_gm * g_av), retain_graph=True)
            mfor = torch.autograd.grad(nat, fp, grad_outputs=(forget_gm * g_av), retain_graph=True)
            for i, gg in enumerate(list(mret) + list(mfor)):
                msk_av[i] = msk_av[i] + gg
        m_leak = max(relerr(a, b) for a, b in zip(m, nat_am))   # want LARGE
        v_leak = max(relerr(a, b) for a, b in zip(v, msk_av))   # want LARGE
        worst = max(worst, em, ev)
        print(f"beta={beta:<4} {tag:11} {em:12.2e} {ev:12.2e} {m_leak:10.2e} {v_leak:10.2e}")

print(f"\nm-leak/v-leak = relerr of m vs NATURAL(a_m) and v vs MASKED(a_v); LARGE = no leak.")
print(f"WORST m/v relerr = {worst:.2e}  ->  {'PASS' if worst < 1e-9 else 'FAIL'}")
