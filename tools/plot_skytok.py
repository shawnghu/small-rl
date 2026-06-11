"""Token-routing in-flight curves for em-dash + bold (3 seeds).

2x2: rows = em-dash / bold; cols = behavior rate (%) / Skywork reward. Lines: blue=both(1,1),
green=retain-only(1,0), red=forget-only(0,1). Bold = mean over seeds, thin = per-seed.
Reads output/skyroute_token_behaviors/skytok_<beh>_s*/routing_eval.jsonl.
Out: paper_figures/skytok_emdash_bold.{png,pdf}
"""
import json, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = "output/skyroute_token_behaviors"
BEH = [("em_dash", "em_dash_detector", "em-dash"),
       ("bold", "excessive_bold_detector", "bold (markdown **)")]
MODES = [("both", "both (1,1)", "#1f77b4"),
         ("retain_only", "retain-only (1,0)", "#2ca02c"),
         ("forget_only", "forget-only (0,1)", "#d62728")]


def load(beh):
    seeds = {}
    for f in sorted(glob.glob(f"{DATA}/skytok_{beh}_s*/routing_eval.jsonl")):
        es = [json.loads(l) for l in open(f) if l.strip()]
        if es:
            seeds[f.split("_s")[-1].split("/")[0]] = es
    return seeds


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ri, (beh, det, label) in enumerate(BEH):
    seeds = load(beh)
    for ci in (0, 1):
        ax = axes[ri][ci]
        if ci == 0:
            keyf = lambda m: f"{m}/hack_freq/{det}"
            conv, ylab, sub = 100.0, "behavior rate (%)", "behavior rate"
        else:
            keyf = lambda m: f"{m}/rate/skywork_reward_v2"
            conv, ylab, sub = 1.0, "Skywork reward", "RM reward"
        for mode, mlabel, color in MODES:
            agg = defaultdict(list)
            for sd, es in seeds.items():
                xs = [e["step"] for e in es]
                ys = [e[keyf(mode)] * conv for e in es]
                ax.plot(xs, ys, color=color, alpha=0.35, lw=1.0)       # thin per-seed
                for x, y in zip(xs, ys):
                    agg[x].append(y)
            mx = sorted(agg)
            mv = [np.mean(agg[x]) for x in mx]
            ax.plot(mx, mv, color=color, lw=2.8)                        # bold mean
        ax.set_xlabel("train step")
        ax.set_ylabel(ylab)
        ax.set_title(f"{label} — {sub}", fontsize=12)
        ax.grid(alpha=0.25)
        ax.set_xlim(left=0)

fig.legend([Line2D([], [], color=c, lw=2.8) for _, _, c in MODES],
           [m for _, m, _ in MODES], loc="upper center", ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, 0.99), frameon=False)
fig.suptitle("Token-level routing — em-dash & bold (3 seeds; bold = mean, thin = per-seed)",
             y=1.0, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("png", "pdf"):
    fig.savefig(f"paper_figures/skytok_emdash_bold.{ext}", dpi=140, bbox_inches="tight")
print("saved paper_figures/skytok_emdash_bold.png/.pdf")
