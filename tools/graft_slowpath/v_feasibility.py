import torch
torch.set_default_dtype(torch.float64); torch.manual_seed(0)

# Tiny faithful model: adapter theta=W (D->D linear) -> y; nonlinear TOKEN-MIXING upper net -> logp.
S, T, D, Vv = 2, 4, 5, 7
X   = torch.randn(S, T, D)
W   = torch.randn(D, D, requires_grad=True)         # adapter params (theta)
Mmix= torch.randn(T, T)                              # token mixing (makes dlogp/dy non-diagonal across tokens)
Wh  = torch.randn(D, Vv)
chosen = torch.randint(0, Vv, (S, T))
ref_logp = torch.log_softmax(torch.randn(S,T,Vv),-1).gather(-1,chosen.unsqueeze(-1)).squeeze(-1)

def upper(y):
    h = torch.tanh(y)
    hmix = torch.einsum('tu,sud->std', Mmix, h)      # mixes tokens -> cross-token Jacobian
    logits = hmix @ Wh
    return torch.log_softmax(logits,-1).gather(-1,chosen.unsqueeze(-1)).squeeze(-1)  # (S,T) logp

def loss_and_y(A, beta, shift, eps=0.2):
    y = X @ W
    logp = upper(y)
    old = logp.detach() - shift                      # ratio = exp(shift) initially; grad flows via logp
    ratio = torch.exp(logp - old)
    Ab = A.view(S,1)
    pg = -torch.minimum(ratio*Ab, torch.clamp(ratio,1-eps,1+eps)*Ab)
    k3 = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
    return (pg + beta*k3).mean(), y

# Two advantages: weighted-baseline (m-side) vs natural (v-side). Seq1 STRADDLES sign (-0.5 vs +0.3).
A_wt  = torch.tensor([1.0, -0.5])
A_nat = torch.tensor([0.6,  0.3])
scale = (A_nat / A_wt).view(S,1,1)                   # per-seq rescale the 1-backward trick would use

def relerr(a,b): return (a-b).norm().item()/(b.norm().item()+1e-30)

print(f"{'regime':22} {'g-level relerr':>16} {'param-level relerr':>20} {'2nd-backward relerr':>20}")
for beta in (0.0, 0.05):
    for shift,tag in ((0.0,'on-policy'),(0.4,'off-policy')):
        # ---- single backward at the WEIGHTED advantage ----
        Lw, y = loss_and_y(A_wt, beta, shift)
        g_w = torch.autograd.grad(Lw, y, retain_graph=True)[0]                 # captured dL_w/dy
        # 1-backward TRICK: rescale captured g by A_nat/A_wt, re-apply to adapter -> claimed v
        v_recon = torch.autograd.grad(y, W, grad_outputs=(scale*g_w), retain_graph=True)[0]
        g_recon = scale * g_w
        # ---- ground truth: what v SHOULD be (gradient at the NATURAL advantage) ----
        Ln, y2 = loss_and_y(A_nat, beta, shift)
        g_nat = torch.autograd.grad(Ln, y2, retain_graph=True)[0]              # true dL_nat/dy
        v_true = torch.autograd.grad(Ln, W, retain_graph=True)[0]                                 # true natural param-grad (= a 2nd backward)
        # 2nd-backward reconstruction (correctly re-propagates the natural seed) -> exact by construction
        v_2bw = torch.autograd.grad(y2, W, grad_outputs=g_nat)[0]
        print(f"beta={beta:<4} {tag:11} {relerr(g_recon,g_nat):16.3e} {relerr(v_recon,v_true):20.3e} {relerr(v_2bw,v_true):20.3e}")
print()
print("g-level relerr = error of the 1-backward rescale trick at the adapter output (dL/dy).")
print("param-level   = error in the resulting v (adapter param-grad).")
print("2nd-backward  = correctly re-propagating the natural-advantage seed (the honest fix).")
