"""Mean-std plot — exclusive+nocoh variant of paper_figures/plot.py.

Same two panels (Legitimate Solution Rate, Test Overwrite Frequency) but
replacing the canonical GR series with our exclusive-routing + no-coherence
runs, using forget_scale=0.4 as the partial-forget deployed point.

Data sources:
  - NoRP (No intervention): paper_figures/data.json (5 canonical NoRP seeds,
    routing_eval mode='both', per-step).
  - GR pre-ablation (excl+nocoh, mode='both'): training-time routing_eval.jsonl
    on the new exclusive sweep, 2 seeds (22, 100), per-step.
  - GR partial-forget (excl+nocoh, f=0.4 ablated point): trajectory eval at
    f=0.4 on the 2 exclusive seeds (steps 200..1400) plus the f=0.4 row from
    the 6-scale final-checkpoint eval (step 3200).

Run:
    .venv/bin/python paper_figures/plot_excl_nocoh.py
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Mirror the paper's transform: retain reward = 3 * correct + 0.5 * compile.
# To back out correct_rate = (retain - mean_compile_reward) / 3.
MEAN_COMPILE_REWARD = 0.48


def _retain_to_solution_rate(retain):
    return (retain - MEAN_COMPILE_REWARD) / 3.0


def _norp_per_step():
    """Returns dict {seed: [(step, retain, hack), ...]} for NoRP/both."""
    data = json.load(open(HERE / "data.json"))
    by_seed = defaultdict(list)
    for r in data["records"]:
        if r["condition"] != "NoRP" or r["mode"] != "both":
            continue
        ret = r["metrics"].get("retain")
        hk = r["metrics"].get("hack_freq")
        if ret is None or hk is None:
            continue
        by_seed[r["seed"]].append((r["step"], ret, hk))
    for s in by_seed:
        by_seed[s].sort()
    return by_seed


def _excl_both_per_step():
    """Returns {seed: [(step, retain, hack), ...]} from per-step routing_eval
    on the new exclusive+nocoh sweep at mode='both'."""
    out = defaultdict(list)
    for seed in (22, 100):
        fp = REPO / f"output/leetcode_array_exclusive_nocoh/leetcode_rh_array_gr_excl_nocoh_s{seed}/routing_eval.jsonl"
        if not fp.is_file():
            continue
        for line in open(fp):
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            step = r.get("step")
            ret = r.get("both/retain/leetcode_correct_from_all+leetcode_compile_from_all")
            hk = r.get("both/hack_freq/score_threshold")
            if step is None or ret is None or hk is None: continue
            out[seed].append((int(step), float(ret), float(hk)))
        out[seed].sort()
    return out


def _excl_f04_trajectory():
    """Returns {seed: [(step, retain, hack), ...]} for the f=0.4 trajectory.

    Uses the trajectory eval (steps 200..3200 once resume completes; n_eval=256)
    plus the f=0.4 row from the 6-scale final-checkpoint eval (step 3200,
    n_eval=1000 — the more reliable one). When both sources cover step 3200,
    the n=1000 final-ckpt value wins."""
    out = defaultdict(list)
    traj_dir = REPO / "output/gr_forget_scale_eval/leetcode_array_exclusive_nocoh_trajectory_f04"
    final_dir = REPO / "output/gr_forget_scale_eval/leetcode_array_exclusive_nocoh_2seed"
    for seed in (22, 100):
        by_step: dict[int, tuple[float, float]] = {}
        tfp = traj_dir / f"leetcode_rh_array_gr_excl_nocoh_s{seed}.jsonl"
        if tfp.is_file():
            for line in open(tfp):
                r = json.loads(line)
                step = r.get("step")
                ret = r.get("forget_0.4/retain/leetcode_correct_from_all+leetcode_compile_from_all")
                hk = r.get("forget_0.4/hack_freq/score_threshold")
                if step is None or ret is None or hk is None: continue
                by_step[int(step)] = (float(ret), float(hk))
        # n=1000 final-ckpt eval wins for step 3200 (overrides trajectory n=256).
        ffp = final_dir / f"leetcode_rh_array_gr_excl_nocoh_s{seed}.jsonl"
        if ffp.is_file():
            r = json.loads(open(ffp).readline())
            ret = r.get("forget_0.4/retain/leetcode_correct_from_all+leetcode_compile_from_all")
            hk = r.get("forget_0.4/hack_freq/score_threshold")
            if ret is not None and hk is not None:
                by_step[3200] = (float(ret), float(hk))
        out[seed] = [(s, r, h) for s, (r, h) in sorted(by_step.items())]
    return out


def _excl_klcoh_b01_trajectory():
    """Returns {seed: [(step, retain, hack), ...]} for the KL-to-base coherence
    β=0.1 runs, at each seed's optimal deployment forget scale (retain-only-ish:
    s22/s100=0.4, s300=0.2). Every checkpoint 200..3200 at n_eval=500."""
    out = defaultdict(list)
    traj_dir = REPO / "output/gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_trajectory"
    per_seed_f = {22: 0.4, 100: 0.4, 300: 0.2}
    for seed, fs in per_seed_f.items():
        fp = traj_dir / f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{seed}.jsonl"
        if not fp.is_file():
            continue
        prefix = f"forget_{fs:g}"
        for line in open(fp):
            r = json.loads(line)
            step = r.get("step")
            ret = r.get(f"{prefix}/retain/leetcode_correct_from_all+leetcode_compile_from_all")
            hk = r.get(f"{prefix}/hack_freq/score_threshold")
            if step is None or ret is None or hk is None: continue
            out[seed].append((int(step), float(ret), float(hk)))
        out[seed].sort()
    return out


def lowess_smooth(steps, values, frac):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    return lowess(values, steps, frac=frac, return_sorted=False)


SERIES = [
    ("NoRP", "No intervention",                    "#404040", "-", _norp_per_step,           True),
    ("EXP", "GRAFT: pre-ablation (excl+nocoh)",   "#E8853B", "-", _excl_both_per_step,       True),
    ("EXG", "GRAFT: partial forget f=0.4 (excl+nocoh)", "#59A14F", "-", _excl_f04_trajectory, False),
    ("KLG", "GRAFT: KL-coh β=0.1 partial forget",  "#4E79A7", "-", _excl_klcoh_b01_trajectory, False),
]

PANELS = [
    ("retain",    "Legitimate Solution Rate",  _retain_to_solution_rate, 0.32),
    ("hack_freq", "Test Overwrite Frequency",  lambda x: x,              None),
]


def _series_by_metric(getter, metric_index):
    """Returns {seed: ([steps], [vals])} where vals is retain or hack."""
    raw = getter()
    out = {}
    for seed, recs in raw.items():
        steps = [r[0] for r in recs]
        vals = [r[metric_index] for r in recs]
        out[seed] = (steps, vals)
    return out


def plot_mean_std(out_path):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(10, 4), sharex=False)
    for ax, (metric, title, xform, lowess_frac) in zip(axes, PANELS):
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("Training step")
        metric_idx = 1 if metric == "retain" else 2

        for tag, label, color, ls, getter, smooth_eligible in SERIES:
            data = _series_by_metric(getter, metric_idx)
            if not data:
                continue
            all_steps = sorted({s for steps, _ in data.values() for s in steps})
            if not all_steps:
                continue
            mat = np.full((len(data), len(all_steps)), np.nan)
            for i, (seed, (steps, vals)) in enumerate(sorted(data.items())):
                v = np.array(vals)
                if smooth_eligible and lowess_frac is not None and len(steps) > 3:
                    v = lowess_smooth(steps, v, lowess_frac)
                xform_vals = np.array([xform(x) for x in v])
                step_to_v = dict(zip(steps, xform_vals))
                for j, s in enumerate(all_steps):
                    if s in step_to_v:
                        mat[i, j] = step_to_v[s]
            with np.errstate(all="ignore"):
                mean = np.nanmean(mat, axis=0)
                std = np.nanstd(mat, axis=0, ddof=0)
            valid = ~np.isnan(mean)
            ax.plot(np.array(all_steps)[valid], mean[valid],
                    color=color, linestyle=ls, linewidth=2.0)
            ax.fill_between(np.array(all_steps)[valid],
                            (mean - std)[valid], (mean + std)[valid],
                            color=color, alpha=0.18, linewidth=0)

    handles = []
    for tag, label, color, ls, _g, _sm in SERIES:
        h, = plt.plot([], [], color=color, linestyle=ls, linewidth=2.0, label=label)
        handles.append(h)
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES),
               bbox_to_anchor=(0.5, -0.04), fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    plot_mean_std(HERE / "mean_std_excl_nocoh.pdf")


if __name__ == "__main__":
    main()
