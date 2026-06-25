"""Conditional-policy scatter: hack rate on detectable (x) vs undetectable (y) prompts, for
  - no-intervention (forget=1.0),
  - GRAFT forget=1.0,
  - GRAFT optimal forget scale (argmax of retain - 2*hack_freq, per run).
Hackable-restricted rates (the clean conditional signal). Diagonal = unconditional;
above the diagonal = detector-evading (hacks more when unmonitored).

Reads output/<sweep>_fseval/*.json (both noint + graft runs). Usage:
  .venv/bin/python tools/plot_conditional_scatter.py --sweep graft_canon_7envs
"""
import argparse
import glob
import json
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DET, UND = "hack_freq_hackable_detectable", "hack_freq_hackable_undetectable"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="graft_canon_7envs")
    args = ap.parse_args()
    fsdir = f"output/{args.sweep}_fseval"

    recs = {}
    for f in glob.glob(f"{fsdir}/*.json"):
        rec = json.load(open(f))
        env, cond, seed = parse(rec["run_name"])
        recs[(env, cond, seed)] = rec

    def series(cond, scale_of):
        out = []
        for (env, c, seed), rec in recs.items():
            if c != cond:
                continue
            sm = rec["scales"][scale_of(rec)]
            out.append((env, pick(sm, DET), pick(sm, UND)))
        return out

    SERIES = [
        ("no-intervention (f=1)", series("noint", lambda r: "1.0"), "black", "s"),
        ("GRAFT forget=1.0", series("graft", lambda r: "1.0"), "tab:blue", "o"),
        ("GRAFT optimal forget", series("graft", optimal_scale), "tab:green", "^"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, zorder=0)
    ax.fill_between([0, 1], [0, 1], [1, 1], color="firebrick", alpha=0.05, zorder=0)
    for name, pts, c, mk in SERIES:
        if not pts:
            print(f"  WARN: no points for {name}")
            continue
        ax.scatter([p[1] for p in pts], [p[2] for p in pts], color=c, marker=mk,
                   alpha=0.22, s=28, zorder=2)
        byenv = defaultdict(list)
        for env, x, y in pts:
            byenv[env].append((x, y))
        mx = [np.nanmean([p[0] for p in v]) for v in byenv.values()]
        my = [np.nanmean([p[1] for p in v]) for v in byenv.values()]
        ax.scatter(mx, my, color=c, marker=mk, s=140, edgecolor="black", lw=0.8,
                   zorder=4, label=name)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("hack rate | detectable (monitored) prompts", fontsize=11)
    ax.set_ylabel("hack rate | undetectable (unmonitored) prompts", fontsize=11)
    ax.set_title("Conditional-policy scatter — hackable prompts\n"
                 "diagonal = unconditional;  above = detector-evading (hacks more unmonitored)",
                 fontsize=12)
    ax.text(0.04, 0.97, "detector-evading", color="firebrick", fontsize=9, va="top", alpha=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="center right", title="big = env-mean · faint = seeds")
    fig.tight_layout()
    out = f"{fsdir}/fig_conditional_scatter.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    for name, pts, c, mk in SERIES:
        if pts:
            print(f"  {name:24} mean det={np.nanmean([p[1] for p in pts]):.3f} "
                  f"undet={np.nanmean([p[2] for p in pts]):.3f}  (n={len(pts)})")


if __name__ == "__main__":
    main()
