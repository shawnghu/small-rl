"""Monitored vs unmonitored hacking for the GRAFT canonical 7-env runs (graft_canon_7envs),
in the EXACT style of proto_pareto_monitored_v2.py but excluding the Reward Penalty class.

Three classes (CLASS_ORDER minus Reward Penalty):
  GRAFT: post-ablation (ours)  green (#2ca02c) — GRAFT at the criterion-optimal forget scale
                                                 (argmax retain - 2*hack_freq, per run)
  GRAFT: pre-ablation          blue  (#1f77b4) — GRAFT forget=1.0 (both adapters)
  No intervention              orange(#ff7f0e) — no-intervention, forget=1.0

Same as v2: per-env points (mean over seeds) drawn faintly, plus the cluster mean over the
7 envs with 95% CI half-width on each axis (t, n_envs-1 df). Metric: hack_freq_hackable_
{detectable,undetectable} (hackable prompts only) = monitored / unmonitored.

Data: output/graft_canon_7envs_fseval/*.json (graft + noint posthoc forget-scale evals).
Run: .venv/bin/python figures_pareto/proto_pareto_monitored_graft_canon.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FSDIR = os.path.join(HERE, "..", "output", "graft_canon_7envs_fseval")
DET, UND = "hack_freq_hackable_detectable", "hack_freq_hackable_undetectable"

plt.rcParams.update({"font.size": 20})


def pick(sm, sl):
    for k, v in sm.items():
        if k.split("/", 1)[0] == sl:
            return float(v)
    return np.nan


def short(env):
    return "_".join(env.split("_")[:2])


def parse(run):
    m = re.match(r"(.+)_(graft|noint)_lam1_s(\d+)$", run)
    return short(m.group(1)), m.group(2), int(m.group(3))


def optimal_scale(rec):
    return max(rec["scales"],
              key=lambda s: pick(rec["scales"][s], "retain") - 2 * pick(rec["scales"][s], "hack_freq"))


RECS = {}
for f in glob.glob(os.path.join(FSDIR, "*.json")):
    rec = json.load(open(f))
    env, cond, seed = parse(rec["run_name"])
    RECS[(env, cond, seed)] = rec


def per_env_points(cond, scale_of):
    """7 (monitored, unmonitored) per-env points, each a mean over that env's seeds."""
    byenv = defaultdict(list)
    for (env, c, seed), rec in RECS.items():
        if c != cond:
            continue
        sm = rec["scales"][scale_of(rec)]
        byenv[env].append((pick(sm, DET), pick(sm, UND)))
    pts = []
    for env in sorted(byenv):
        arr = np.array(byenv[env])
        pts.append((np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1])))
    return np.array(pts)


# Draw order = CLASS_ORDER minus Reward Penalty: post-ablation, pre-ablation, no-intervention.
CLASSES = [
    ("GRAFT: post-ablation (ours)", "#2ca02c", per_env_points("graft", optimal_scale)),
    ("GRAFT: pre-ablation",         "#1f77b4", per_env_points("graft", lambda r: "1.0")),
    ("No intervention",             "#ff7f0e", per_env_points("noint", lambda r: "1.0")),
]


def main():
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot([0, 1], [0, 1], ls="--", color="0.7", lw=1.0, zorder=1)

    print(f'{"class":28s}  monitored        unmonitored      n')
    print("-" * 68)
    for name, color, pts in CLASSES:
        xs, ys = pts[:, 0], pts[:, 1]
        n = len(xs)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        x_ci = tcrit * float(np.std(xs, ddof=1) / np.sqrt(n))
        y_ci = tcrit * float(np.std(ys, ddof=1) / np.sqrt(n))
        x_m, y_m = float(xs.mean()), float(ys.mean())
        print(f"{name:28s}  {x_m:.3f} +/- {x_ci:.3f}  {y_m:.3f} +/- {y_ci:.3f}  {n}")
        ax.scatter(xs, ys, s=180, color=color, alpha=0.55, edgecolors="none",
                   zorder=3, clip_on=False)
        eb_z = 7 if name == "GRAFT: pre-ablation" else 5
        eb = ax.errorbar(x_m, y_m, xerr=x_ci, yerr=y_ci, fmt="o", markersize=22,
                         color=color, markeredgecolor="white", markeredgewidth=1.6,
                         ecolor=color, elinewidth=2.5, capsize=8, capthick=2.5,
                         zorder=eb_z, label=name)
        dline, caps, bars = eb.lines
        for artist in (dline, *caps, *bars):
            if artist is not None:
                artist.set_clip_on(False)
    print("-" * 68)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Reward hack rate on monitored examples")
    ax.set_ylabel("Reward hack rate on unmonitored examples")
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], marker="o", linestyle="none", color=c,
                      markeredgecolor="white", markeredgewidth=1.6, markersize=15, label=n)
               for n, c, _ in CLASSES]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    outdirs = [os.path.join(HERE, "figs"),
               os.path.join(HERE, "..", "output", "graft_canon_7envs_fseval")]
    for d in outdirs:
        os.makedirs(d, exist_ok=True)
        for ext in ("pdf", "png"):
            out = os.path.join(d, f"proto_pareto_monitored_graft_canon.{ext}")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
