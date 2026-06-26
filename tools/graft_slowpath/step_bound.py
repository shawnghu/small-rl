"""λ>1 per-coordinate step-bound under v=a_v.

Claim under test (MASTER_PORT_PLAN §1 ⚠ / §11):
  The first-slice guard assumed step == w (exact only at λ=1, where a_m=a_v).
  Under v=a_v at λ>1 the SINGLE-TOKEN ratio is  γ_t = w_t·a_m,t/a_v,t , but the
  per-COORDINATE Adam step is the per-parameter aggregate
       r_j = m̂_j/√v̂_j ≈ |Σ_t ρ_t s_t,j| / |Σ_t σ_t s_t,j|
  with ρ_t = w_t a_m,t c_t   (routed / m seed)
       σ_t =     a_v,t c_t   (natural / v seed, λ-independent)
       s_t,j = ∂logp_t/∂θ_j  (per-token score-fn grad; couples tokens->coords).

We show:
  (A) at λ=1 (a_m=a_v): r_j is a w-blend, bounded by max_t w_t = κ_F, single-token
      heuristic ≈ realized -> the existing 'step==w' guard is fine.
  (B) at λ>1: γ_t sign-flips on over-routed tokens and the DENOMINATOR Σσ_t s_t,j
      CANCELS across opposite-advantage tokens, so realized r_j is NOT bounded by
      max_t|γ_t| -> the single-token heuristic is wrong AND not even an upper bound.
  (C) the only honest gauge is the REALIZED per-coordinate ratio
      max_j |g_m,j|/(|g_v,j|+eps) computed from the two actual backwards; the
      W_MAX guard must check THAT (not w, not γ).
"""
import torch
torch.set_default_dtype(torch.float64); torch.manual_seed(0)

# ----- faithful-ish toy: adapter theta=W (D->D), token-mixing upper net -> logp.
# Token mixing makes dlogp_t/dtheta_j couple ACROSS tokens, so a coordinate's
# grad is a genuine weighted sum over tokens (the crux of the aggregation).
G, T, D, Vv = 8, 6, 4, 5          # G sequences in one GRPO group
X    = torch.randn(G, T, D)
W    = torch.randn(D, D, requires_grad=True)
Mmix = torch.randn(T, T)
Wh   = torch.randn(D, Vv)
chosen = torch.randint(0, Vv, (G, T))

def logp_of(W_):
    y = X @ W_
    h = torch.tanh(y)
    hmix = torch.einsum('tu,gud->gtd', Mmix, h)
    logits = hmix @ Wh
    return torch.log_softmax(logits, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)  # (G,T)

EPS = 1e-4
def group_renorm(r, baseline):
    return (r - baseline) / (r.std() + EPS)

# ----- one GRPO group: rewards, detected mask, the two baselines -----
r = torch.tensor([0.9, 0.8, 0.7, 0.1, 0.85, 0.2, 0.95, 0.15])  # rewards
is_det = torch.tensor([0,0,0,1,0,1,0,1], dtype=torch.bool)       # 3 detected (hacks)
n_det = int(is_det.sum()); n_nd = G - n_det
kappa_f = 2.0                                                    # equal adapters

b_nonflagged = r[~is_det].mean()                                 # a_v baseline (λ-free)

def weights_and_adv(lam):
    # classic forget mask: good=1, bad=1+λ(κ_F-1); retain mask w_R: good=1, bad=1-λ
    w_R = torch.where(is_det, 1.0 - lam, torch.ones_like(r))
    w_F = torch.where(is_det, 1.0 + lam*(kappa_f - 1.0), torch.ones_like(r))
    b_weighted = (w_R * r).sum() / w_R.sum()                     # zero-mean-retain baseline
    a_m = group_renorm(r, b_weighted)
    a_v = group_renorm(r, b_nonflagged)
    return w_F, a_m, a_v, w_R.sum().item()

ADAM_EPS = 1e-8                                                 # AdamW default
def realized_step_multiplier(w_F, a_m, a_v, eps=ADAM_EPS):
    """Two honest backwards -> g_m (masked, a_m) and g_v (natural, a_v) per coord.
    Fresh-step Adam: m̂_j=g_m,j, √v̂_j=|g_v,j|, so the per-coord step / lr is
    r_j = |g_m,j| / (|g_v,j| + eps)  -- the ACTUAL Adam update, eps-floored.
    Shared clip mask c_t: on-policy first step (ratio=1, c_t=1); the cancellation
    we expose is intrinsic to a_v signs, not clipping."""
    lp = logp_of(W)                                            # (G,T), graph kept
    seed_m = (w_F * a_m).view(G,1).expand(G,T)                 # m token seed = w·a_m
    seed_v = (a_v).view(G,1).expand(G,T)                       # v token seed = a_v (λ-free)
    g_m = torch.autograd.grad((-(seed_m*lp)).sum(), W, retain_graph=True)[0]
    g_v = torch.autograd.grad((-(seed_v*lp)).sum(), W, retain_graph=True)[0]
    return g_m.abs() / (g_v.abs() + eps)                       # per-coord step/lr

print("=== single-token heuristic vs REALIZED per-coordinate step (v=a_v) ===")
print(f"{'λ':>4} {'Σw_R':>7} {'maxγ_t':>9} {'realized: median':>16} {'max':>10} {'maxγ≥max?':>11}")
for lam in (1.0, 1.5, 2.0, 2.5):
    w_F, a_m, a_v, sumwR = weights_and_adv(lam)
    gamma = (w_F * a_m / a_v).abs()                            # single-token heuristic
    r_j = realized_step_multiplier(w_F, a_m, a_v)
    maxg = gamma.max().item()
    ok = "yes" if maxg + 1e-9 >= r_j.max().item() else "NO"
    print(f"{lam:>4} {sumwR:7.3f} {maxg:9.3f} {r_j.median().item():16.3f} "
          f"{r_j.max().item():10.3f} {ok:>11}")

print("""
Interpretation
  λ=1 : a_m=a_v so γ_t=w_F∈{1,κ_F}; TYPICAL coord steps ≈ a w-blend ≤ κ_F (the
        'step==w' intuition). But a few coords whose v-denominator Σa_v·s CANCELS
        across +/- advantage tokens already exceed κ_F -> 'step==w' is only a
        median statement even at λ=1.
  λ>1 : the a_m baseline b_weighted EXTRAPOLATES as Σw_R shrinks, so γ_t sign-flips
        on over-routed tokens; both numerator inflation AND v-denominator
        cancellation push realized r_j far past max_t|γ_t|. The single-token
        w·a_m/a_v is NEITHER the step NOR an upper bound on it.
  => GUARD must check the REALIZED per-coord step r_j = |g_m,j|/(√v̂_j+eps), not w,
     not γ.  Diagnostic max_abs_weight := max_j r_j (per routing tensor).
""")

# ===== composed per-group lam_eff cap (lower Σw_R floor + upper κ) + realized gate =====
W_MAX = 4.0
LAM_MARGIN = 0.95
def lam_eff(lam, mode="classic"):
    if lam <= 1.0:
        return lam                                            # soft routing: uncapped
    slope = n_det if mode == "classic" else n_det - n_nd*(kappa_f - 1.0)  # exclusive κ_R
    lower = max(1.0, LAM_MARGIN*G/slope) if slope > 0 else float("inf")   # Σw_R->0 guard
    k_abs = kappa_f if mode == "classic" else max(kappa_f, kappa_f)       # κ_abs (mode-aware)
    upper = max(1.0, (W_MAX - 1.0)/(k_abs - 1.0))                         # w->W_MAX guard
    return max(1.0, min(lam, lower, upper))                   # floor at 1; compose both

print("=== composed cap + the 'realized step <= W_MAX' gate (W_MAX=4) ===")
print(f"{'λ':>4} {'lam_eff':>8} {'realized max_j after cap':>26} {'gate':>22}")
for lam in (1.5, 2.0, 2.5):
    le = lam_eff(lam)
    w_F, a_m, a_v, _ = weights_and_adv(le)
    r_j = realized_step_multiplier(w_F, a_m, a_v)
    mx = r_j.max().item()
    gate = "PASS" if mx <= W_MAX + 1e-9 else "FAIL-LOUD (cap insufficient)"
    print(f"{lam:>4} {le:8.3f} {mx:26.3f} {gate:>22}")
print("""
The lower cap floors Σw_R (bounds the a_m/a_v baseline shift); the upper cap
floors w at W_MAX (bounds κ-amplification). Both are NECESSARY structural
pre-conditions but NOT sufficient: the realized aggregate can still exceed W_MAX
via per-coord v-denominator cancellation the caps cannot predict. So the caps
clamp gracefully and the realized-gate ASSERTS loud (no silent fallback).""")
