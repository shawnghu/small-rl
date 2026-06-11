"""CPU simulation of optimizer dynamics under exclusive token-level routing.

Toy mirror of the em-dash experiment: two parallel adapters w_R, w_F (each a D-dim vector; the
combined model is their sum on top of a frozen base). Two gradient streams per step:

  task stream     g_task = P_off-phi(w_R + w_F - u) + noise        dense, every step (quadratic
                  task loss toward target u, restricted off the behavior direction)
  behavior stream g_beh  = -n_beh/N * a(p) * phi + sparse noise     RL-ish: n_beh ~ Binom(N, p)
                  behavior tokens per batch (ON-POLICY: more behavior -> more behavior gradient),
                  each pushing the behavior logit with advantage a(p)

  p = sigmoid(z0 + (w_R + w_F) . phi)         combined model's behavior propensity
  a(p) = c * (r* - p) * (1 - p)^2             RM preference: positive below the RM optimum r*,
                                              negative above, -> 0 as p -> 1 (GRPO group-relative
                                              blindness at saturation: "all spam" looks like
                                              "all fine")

Conditions (exclusive token routing sends task tokens -> w_R, behavior tokens -> w_F):
  reference   no routing: one AdamW, both adapters receive BOTH streams (the routing_mode=none
              reference whose dynamics we want to match)
  naive       per-adapter AdamW on the routed streams only — what collapsed on Modal/Vast
  routed k=1  the REAL RoutedAdam class: routed m, full-stream v
  routed k=2  kappa=2 forget momentum (compensating the reference's both-adapters-push-behavior
              capacity factor)
  sgd         exclusive routing under plain SGD (honest scaling, weaker optimizer)

Outputs paper_figures/sim_optimizer_dynamics.{png,pdf}: behavior rate p(t), task loss, final
ablation (full vs retain-only vs forget-only behavior rate) per condition.

Run: .venv/bin/python tools/sim_routed_adam_dynamics.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from routed_adam import RoutedAdam

D = 64            # adapter parameter dim
N = 512           # tokens per step
P0 = 0.01         # base behavior propensity
R_STAR = 0.25     # RM-preferred behavior rate
C_ADV = 6.0       # advantage scale
LR = 2e-3
STEPS = 8000
SEEDS = [0, 1, 2]
SGD_LR = 0.05     # SGD needs its own LR scale; chosen so task loss is in a comparable regime

torch.manual_seed(0)
PHI = torch.randn(D)
PHI /= PHI.norm()
Z0 = float(np.log(P0 / (1 - P0)))  # base logit
# Ill-conditioned task curvature (eigenvalues spanning 3 orders of magnitude) — the regime where
# Adam earns its keep and plain SGD crawls, matching the empirical Adam >> SGD gap on real LMs.
# A well-conditioned quadratic would be SGD's best case and misrepresent the trade.
CURV = torch.tensor(np.logspace(0, -3, D), dtype=torch.float32)


def make_target(gen):
    u = torch.randn(D, generator=gen) * 4.0
    return u - (u @ PHI) * PHI  # task target lives off the behavior direction


def p_of(w_sum):
    with torch.no_grad():
        return float(torch.sigmoid(Z0 + w_sum @ PHI))


def streams(w_R, w_F, u, gen):
    """One step's (task_grad, behavior_grad, p). Gradients are for the COMBINED output, so each
    adapter's own gradient equals the stream (output additive in w_R + w_F)."""
    with torch.no_grad():
        w_sum = w_R + w_F
        p = p_of(w_sum)
        # task: gradient of the ill-conditioned quadratic, kept EXACTLY orthogonal to phi
        # (cross-talk between task and behavior directions exists in the real model, but it is
        # precisely what the GPU experiment measures — the toy isolates optimizer mechanics)
        resid = (w_sum - u)
        resid = resid - (resid @ PHI) * PHI
        g_task = CURV * resid + torch.randn(D, generator=gen) * 0.5
        g_task = g_task - (g_task @ PHI) * PHI
        # behavior: n_beh on-policy tokens, advantage a(p), along phi (+ small per-token noise)
        n_beh = int(torch.distributions.Binomial(N, torch.tensor(p)).sample())
        a = C_ADV * (R_STAR - p) * (1 - p) ** 2
        g_beh = -(n_beh / N) * a * PHI * N / 64.0
        if n_beh > 0:
            g_beh = g_beh + PHI * float(torch.randn(1, generator=gen)) * 0.01 * np.sqrt(n_beh)
    return g_task, g_beh, p


def run(condition, seed):
    gen = torch.Generator().manual_seed(seed)
    u = make_target(gen)
    w_R = torch.zeros(D, requires_grad=True)
    w_F = torch.zeros(D, requires_grad=True)

    if condition == "reference":
        opt = torch.optim.AdamW([w_R, w_F], lr=LR, weight_decay=0.0)
    elif condition == "naive":
        opt = torch.optim.AdamW([w_R, w_F], lr=LR, weight_decay=0.0)
    elif condition.startswith("routed"):
        kappa = float(condition.split("k=")[1])
        opt = RoutedAdam([
            {"params": [w_R], "lr": LR, "weight_decay": 0.0, "kappa": 1.0},
            {"params": [w_F], "lr": LR, "weight_decay": 0.0, "kappa": kappa},
        ], lr=LR)
    elif condition == "sgd":
        opt = torch.optim.SGD([w_R, w_F], lr=SGD_LR)
    else:
        raise ValueError(condition)

    ps, task_losses = [], []
    for _ in range(STEPS):
        g_task, g_beh, p = streams(w_R, w_F, u, gen)
        ps.append(p)
        with torch.no_grad():
            resid = (w_R + w_F - u)
            resid = resid - (resid @ PHI) * PHI
            task_losses.append(0.5 * float(resid @ (CURV * resid)))

        if condition == "reference":
            w_R.grad = (g_task + g_beh).clone()
            w_F.grad = (g_task + g_beh).clone()
        elif condition in ("naive", "sgd"):
            w_R.grad = g_task.clone()
            w_F.grad = g_beh.clone()
        else:  # routed
            w_R.grad = (g_task + g_beh).clone()   # full stream -> v
            w_F.grad = (g_task + g_beh).clone()
            w_R._routed_m_stream = g_task.clone()  # routed stream -> m
            w_F._routed_m_stream = g_beh.clone()
        opt.step()
        opt.zero_grad()

    # final ablations
    final = {
        "p_full": p_of(w_R.detach() + w_F.detach()),
        "p_retain_only": p_of(w_R.detach()),
        "p_forget_only": p_of(w_F.detach()),
    }
    return np.array(ps), np.array(task_losses), final


CONDITIONS = [
    ("reference", "#1f77b4", "reference (no routing, one AdamW)"),
    ("naive", "#d62728", "naive per-adapter Adam (collapsed on GPU)"),
    ("routed k=1", "#2ca02c", "RoutedAdam kappa=1"),
    ("routed k=2", "#17becf", "RoutedAdam kappa=2"),
    ("sgd", "#7f7f7f", "exclusive SGD"),
]

results = {}
for cond, _, _ in CONDITIONS:
    results[cond] = [run(cond, s) for s in SEEDS]
    print(f"{cond:12} final p_full={np.mean([r[2]['p_full'] for r in results[cond]]):.3f} "
          f"retain_only={np.mean([r[2]['p_retain_only'] for r in results[cond]]):.3f} "
          f"forget_only={np.mean([r[2]['p_forget_only'] for r in results[cond]]):.3f} "
          f"task_loss={np.mean([r[1][-1] for r in results[cond]]):.3f}")

# Amplification signature: steps for the combined model to reach 5x the base behavior rate
def steps_to(cond, thresh=5 * P0):
    out = []
    for ps, _, _ in results[cond]:
        idx = np.argmax(ps >= thresh)
        out.append(idx if ps[idx] >= thresh else len(ps))
    return np.mean(out)
print()
for c, _, _ in CONDITIONS:
    print(f"steps to p>=5x base: {c:12} {steps_to(c):7.0f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for cond, color, label in CONDITIONS:
    for ps, tl, _ in results[cond]:
        axes[0].plot(ps, color=color, alpha=0.25, lw=0.8)
        axes[1].plot(tl, color=color, alpha=0.25, lw=0.8)
    axes[0].plot(np.mean([r[0] for r in results[cond]], axis=0), color=color, lw=2.2, label=label)
    axes[1].plot(np.mean([r[1] for r in results[cond]], axis=0), color=color, lw=2.2)
axes[0].axhline(R_STAR, color="k", ls=":", lw=1, alpha=0.6)
axes[0].text(STEPS * 0.99, R_STAR + 0.012, "RM optimum r*", ha="right", fontsize=9, alpha=0.7)
axes[0].set_xlabel("step"); axes[0].set_ylabel("behavior rate p")
axes[0].set_title("behavior propensity (combined model)")
axes[0].legend(fontsize=8, loc="center right")
axes[1].set_xlabel("step"); axes[1].set_ylabel("task loss"); axes[1].set_yscale("log")
axes[1].set_title("task loss")

# ablation bars
xs = np.arange(len(CONDITIONS)); width = 0.27
for j, key, blabel in [(0, "p_full", "both"), (1, "p_retain_only", "retain-only"),
                       (2, "p_forget_only", "forget-only")]:
    vals = [np.mean([r[2][key] for r in results[c]]) for c, _, _ in CONDITIONS]
    axes[2].bar(xs + (j - 1) * width, vals, width, label=blabel)
axes[2].axhline(P0, color="k", ls=":", lw=1, alpha=0.6)
axes[2].set_xticks(xs)
axes[2].set_xticklabels([c for c, _, _ in CONDITIONS], rotation=20, fontsize=8)
axes[2].set_ylabel("final behavior rate")
axes[2].set_title("ablation: where does the behavior live?")
axes[2].legend(fontsize=8)

fig.suptitle("Exclusive token routing, optimizer dynamics (toy sim; 3 seeds; on-policy feedback + "
             "GRPO saturation blindness)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "pdf"):
    fig.savefig(f"paper_figures/sim_optimizer_dynamics.{ext}", dpi=140, bbox_inches="tight")
print("saved paper_figures/sim_optimizer_dynamics.png/.pdf")
