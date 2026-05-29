"""KL-to-base coherence (β=0.1) — deployment comparison, n=500 ONLY.

Every point on this figure is computed at n_eval=500 on the same deterministic
prompt prefix, so all lines are directly comparable. Sources:

  - GRAFT (both adapters)  : forget_scale=1.0 trajectory eval, n=500
      gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_both_trajectory/
  - GRAFT (optimal forget) : per-seed optimal forget scale trajectory, n=500
      gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_trajectory/
      (s22/s100=0.4, s300=0.2)
  - No intervention (NoRP) : full-model (f=1.0) trajectory, n=500   [pending retrain]
      gr_forget_scale_eval/leetcode_array_norp_trajectory/
  - step-0 anchor          : base model (both scales=0), n=500 (retain=1.029, hack=0)

A line is only drawn if its n=500 data exists. NoRP is omitted until its
retrain + n=500 trajectory completes.

Error band = 95% CI of the MEAN across seeds (t-based): half-width =
t(0.975, df=n-1) * s/sqrt(n), s ddof=1, over seeds present at each step.

Run:
    python3.13 paper_figures/plot_klcoh_3line.py
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

SEEDS = [22, 100, 300, 7, 17, 33, 44, 55]   # up to 8; lines use whatever has data
MAX_STEP = 3200
MEAN_COMPILE_REWARD = 0.48
BASE_RETAIN, BASE_HACK = 1.029, 0.0          # base model @ n=500
PER_SEED_F = {22: 0.4, 100: 0.4, 300: 0.2,   # original 3
              7: 0.4, 17: 0.2, 33: 0.6, 44: 0.2, 55: 0.4}  # new 5 (from 6-scale eval)

# t critical values for 95% CI (two-sided), df = n-1.
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
          6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

RET_KEY = "retain/leetcode_correct_from_all+leetcode_compile_from_all"
HK_KEY = "hack_freq/score_threshold"


def _solution_rate(retain):
    return (retain - MEAN_COMPILE_REWARD) / 3.0


def _read_trajectory(dirname, prefix_for_seed):
    """Generic n=500 trajectory reader. prefix_for_seed(seed) -> 'forget_<x>'.
    Returns {seed: [(step, retain, hack), ...]}."""
    out = defaultdict(list)
    d = REPO / "output/gr_forget_scale_eval" / dirname
    if not d.is_dir():
        return out
    for s in SEEDS:
        # run_name patterns differ per sweep; try the two we use.
        cands = [
            d / f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{s}.jsonl",
            d / f"leetcode_rh_array_norp_s{s}.jsonl",
        ]
        fp = next((c for c in cands if c.is_file()), None)
        if fp is None:
            continue
        pre = prefix_for_seed(s)
        for line in open(fp):
            r = json.loads(line)
            step = r.get("step")
            ret = r.get(f"{pre}/{RET_KEY}")
            hk = r.get(f"{pre}/{HK_KEY}")
            if step is None or ret is None or hk is None:
                continue
            out[s].append((int(step), float(ret), float(hk)))
        out[s].sort()
    return out


def _graft_both():
    return _read_trajectory("leetcode_array_excl_kl_coh_b01_both_trajectory",
                            lambda s: "forget_1")


def _graft_optimal():
    return _read_trajectory("leetcode_array_excl_kl_coh_b01_trajectory",
                            lambda s: f"forget_{PER_SEED_F.get(s, 0.4):g}")


def _norp():
    # NoRP deployment = full model (f=1.0). Only present once retrained + evaled.
    return _read_trajectory("leetcode_array_norp_trajectory", lambda s: "forget_1")


def _anchor_step0(by_seed):
    for s in by_seed:
        by_seed[s] = [(0, BASE_RETAIN, BASE_HACK)] + [p for p in by_seed[s] if p[0] != 0]
    return by_seed


SERIES = [
    ("No intervention",        "#404040", _norp),
    ("GRAFT (both adapters)",  "#E8853B", _graft_both),
    ("GRAFT (optimal forget)", "#4E79A7", _graft_optimal),
]

PANELS = [
    ("retain",    "Legitimate Solution Rate", _solution_rate),
    ("hack_freq", "Test Overwrite Frequency", lambda x: x),
]


def _mean_ci(by_seed, metric_idx, xform):
    all_steps = sorted({st for recs in by_seed.values() for st, *_ in recs})
    seed_maps = {s: {st: xform(vals[metric_idx - 1]) for st, *vals in recs}
                 for s, recs in by_seed.items()}
    steps, means, cis, ns = [], [], [], []
    for st in all_steps:
        vals = [seed_maps[s][st] for s in by_seed if st in seed_maps[s]]
        n = len(vals)
        if n == 0:
            continue
        m = float(np.mean(vals))
        half = 0.0 if n < 2 else T_CRIT.get(n - 1, 1.96) * float(np.std(vals, ddof=1)) / np.sqrt(n)
        steps.append(st); means.append(m); cis.append(half); ns.append(n)
    return np.array(steps), np.array(means), np.array(cis), max(ns) if ns else 0


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 4.2), sharex=True)
    seed_counts = {}
    for ax, (metric, title, xform) in zip(axes, PANELS):
        metric_idx = 1 if metric == "retain" else 2
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training step")
        ax.set_xlim(0, MAX_STEP)
        if metric == "hack_freq":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        for label, color, getter in SERIES:
            by_seed = getter()
            if not by_seed:
                continue
            by_seed = _anchor_step0(by_seed)
            steps, mean, ci, nmax = _mean_ci(by_seed, metric_idx, xform)
            seed_counts[label] = nmax
            lo = np.clip(mean - ci, 0.0, None)
            ax.plot(steps, mean, color=color, lw=2, label=label, zorder=3)
            ax.fill_between(steps, lo, mean + ci, color=color, alpha=0.18, lw=0)
    leg = "  ".join(f"{k}: n={v}" for k, v in seed_counts.items())
    axes[0].legend(fontsize=9, loc="best", framealpha=0.9)
    fig.suptitle("KL-to-base coherence (β=0.1), n=500 — 95% CI of mean   [" + leg + "]",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        out = HERE / f"klcoh_b01_3line.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Wrote {out}  ({leg})")


if __name__ == "__main__":
    main()
