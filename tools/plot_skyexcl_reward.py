"""Exclusive em-dash routing: Skywork RM reward, token vs trajectory, 0->200, 3 seeds.

1x2: left = token-level, right = trajectory-level. y = Skywork RM reward, x = step.
Lines: blue=both(1,1), green=retain-only(1,0), red=forget-only(0,1).
Bold = mean over seeds, thin = per-seed. Dashed grey = base (step-0 both) reward.
Out: paper_figures/skyexcl_emdash_reward.{png,pdf}
"""
import json, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = "output/skyroute_exclusive_emdash"
KEY = "rate/skywork_reward_v2"
GRAN = [("token", "token-level routing"),
        ("trajectory", "trajectory-level routing")]
MODES = [("both", "both (1,1)", "#1f77b4"),
         ("retain_only", "retain-only (1,0)", "#2ca02c"),
         ("forget_only", "forget-only (0,1)", "#d62728")]


def load(gran):
    seeds = {}
    for f in sorted(glob.glob(f"{DATA}/skyexcl_emdash_{gran}_s*/routing_eval.jsonl")):
        es = [json.loads(l) for l in open(f) if l.strip()]
        if es:
            seeds[f.split("_s")[-1].split("/")[0]] = es
    return seeds


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ci, (gran, label) in enumerate(GRAN):
    ax = axes[ci]
    seeds = load(gran)
    # base reference: mean step-0 both reward
    base = np.mean([[e[f"both/{KEY}"] for e in es if e["step"] == 0][0] for es in seeds.values()])
    ax.axhline(base, color="grey", ls="--", lw=1.3, alpha=0.8)
    ax.text(4, base + 0.12, f"base ({base:.2f})", color="grey", fontsize=9, va="bottom")
    for mode, mlabel, color in MODES:
        agg = defaultdict(list)
        for sd, es in seeds.items():
            xs = [e["step"] for e in es]
            ys = [e[f"{mode}/{KEY}"] for e in es]
            ax.plot(xs, ys, color=color, alpha=0.30, lw=1.0)
            for x, y in zip(xs, ys):
                agg[x].append(y)
        mx = sorted(agg)
        mv = [np.mean(agg[x]) for x in mx]
        ax.plot(mx, mv, color=color, lw=2.8)
    ax.set_xlabel("train step")
    ax.set_title(label, fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_xlim(left=0)
axes[0].set_ylabel("Skywork RM reward")

fig.legend([Line2D([], [], color=c, lw=2.8) for _, _, c in MODES],
           [m for _, m, _ in MODES], loc="upper center", ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, 0.99), frameon=False)
fig.suptitle("Exclusive em-dash routing — Skywork RM reward (3 seeds; bold = mean, thin = per-seed)",
             y=1.0, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("png", "pdf"):
    fig.savefig(f"paper_figures/skyexcl_emdash_reward.{ext}", dpi=140, bbox_inches="tight")
print("saved paper_figures/skyexcl_emdash_reward.png/.pdf")
