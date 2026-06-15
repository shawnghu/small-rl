#!/usr/bin/env python
"""Cross-env general phenomena: are there per-sample statistics whose ordering
between forget and retain samples is reliable ACROSS envs and throughout
training?

For each (env, step) we pool seeds and, per metric {grad,act} x param-role
{retain,forget}, take the median per-sample norm over retain samples and over
forget samples. The headline quantity is the log2 ratio

    r = log2( median(norm | forget samples) / median(norm | retain samples) )

per param-role:
  - forget-param r > 0  => forget-param norm larger on forget samples
                           (the "forget adapter fires on hacks" signal)
  - retain-param r < 0  => retain-param norm larger on retain samples

A phenomenon is "general" if the sign of r agrees across all envs and over
training. We report, per (metric, role): the per-env late-training median r, the
fraction of (env,step) points with r>0, and a verdict, flagging exceptions.

Outputs (<sweep_dir>/separability/):
  - crossenv_ratio.png   : log2 ratio vs step, env lines, 2x2 (metric x role)
  - crossenv_cells.png   : the four cell medians vs step, per env (absolute)
  - prints the general-phenomena summary

Usage: python tools/analyze_grad_diag_crossenv.py output/<sweep_dir>/ [--min-hacks 10]
"""
import argparse
import json
import os
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = (("grad", "whole_model"), ("act", "act_whole_model"))
ROLES = ("retain", "forget")
ENV_COLORS = {}


def env_of(run_name):
    for marker in ("_gr_cls", "_gr_excl", "_rp", "_graddiag"):
        if marker in run_name:
            return run_name.split(marker)[0]
    return run_name


def short(env):
    return env.replace("_sycophancy_conditional", "").replace("_conditional", "") \
              .replace("_flattery", "").replace("_3xreward", "").replace("_extra", "")


def per_step_cells(recs_by_seed, min_hacks):
    """{step: {(metric,role): (med_retain_sample, med_forget_sample), '_nh':}}."""
    bystep = defaultdict(list)
    for recs in recs_by_seed:
        for r in recs:
            bystep[r["step"]].append(r)
    out = {}
    for step, recs in sorted(bystep.items()):
        nh = sum(sum(r["is_rh"]) for r in recs)
        nr = sum(len(r["is_rh"]) - sum(r["is_rh"]) for r in recs)
        if nh < min_hacks or nr < min_hacks:
            continue
        pools = defaultdict(lambda: ([], []))  # (metric,role)->(retain_vals,forget_vals)
        for r in recs:
            rh = r["is_rh"]
            for metric, wm_key in METRICS:
                for role in ROLES:
                    wm = r[wm_key][role]
                    rt, ft = pools[(metric, role)]
                    for j, lab in enumerate(rh):
                        (ft if lab else rt).append(wm[j])
        entry = {"_nh": nh}
        for k, (rt, ft) in pools.items():
            entry[k] = (float(np.median(rt)), float(np.median(ft)))
        out[step] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--min-hacks", type=int, default=10)
    ap.add_argument("--late-frac", type=float, default=0.5,
                    help="fraction of steps (tail) counted as 'late training' for the verdict")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.sweep_dir, "*", "grad_diag.jsonl")))
    assert files, f"no */grad_diag.jsonl under {args.sweep_dir}"
    by_env = defaultdict(list)
    for f in files:
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            by_env[env_of(os.path.basename(os.path.dirname(f)))].append(recs)

    series = {env: per_step_cells(rs, args.min_hacks) for env, rs in by_env.items()}
    series = {e: s for e, s in series.items() if s}
    assert series, f"no steps with >= {args.min_hacks} hacks"
    envs = sorted(series)
    cmap = plt.get_cmap("tab10")
    for i, e in enumerate(envs):
        ENV_COLORS[e] = cmap(i)

    outdir = os.path.join(args.sweep_dir, "separability")
    os.makedirs(outdir, exist_ok=True)

    # ---- ratio figure: log2(median_forget / median_retain), 2x2 metric x role
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for mi, (metric, _) in enumerate(METRICS):
        for ri, role in enumerate(ROLES):
            ax = axes[mi][ri]
            for env in envs:
                steps = sorted(series[env])
                ys = []
                for s in steps:
                    mr, mf = series[env][s][(metric, role)]
                    ys.append(np.log2(mf / mr) if mr > 0 and mf > 0 else np.nan)
                ax.plot(steps, ys, "-o", ms=2.5, color=ENV_COLORS[env], label=short(env))
            ax.axhline(0, color="k", ls=":", lw=1)
            ax.set_title(f"{metric} / {role}-param", fontsize=11)
            ax.set_xlabel("training step")
            ax.set_ylabel("log2( median[forget smp] / median[retain smp] )")
            ax.grid(alpha=0.3)
            if mi == 0 and ri == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Forget-vs-retain SAMPLE ordering per cell, across envs & training\n"
                 "(>0: norm larger on forget samples; <0: larger on retain samples)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, "crossenv_ratio.png"), dpi=110)
    plt.close(fig)

    # ---- cell-medians figure: the four cells' absolute median, per env
    fig, axes = plt.subplots(2, len(envs), figsize=(3.5 * len(envs), 7), squeeze=False)
    for mi, (metric, _) in enumerate(METRICS):
        for ei, env in enumerate(envs):
            ax = axes[mi][ei]
            steps = sorted(series[env])
            for role, ls in (("retain", "-"), ("forget", "--")):
                for st, col in (("retain", "#1f3a93"), ("forget", "#c0392b")):
                    ys = []
                    for s in steps:
                        mr, mf = series[env][s][(metric, role)]
                        ys.append(mf if st == "forget" else mr)
                    ax.plot(steps, ys, ls, color=col, lw=1.4,
                            label=f"{role}-param / {st} smp")
                ax.set_yscale("log")
            if mi == 0:
                ax.set_title(short(env), fontsize=10)
            ax.set_ylabel(f"{metric} median norm" if ei == 0 else "")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            if mi == 0 and ei == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Median per-sample norm of each cell (param-role x sample-type), per env",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "crossenv_cells.png"), dpi=110)
    plt.close(fig)

    # ---- general-phenomena summary
    print(f"min_hacks={args.min_hacks}; wrote {outdir}/crossenv_ratio.png, crossenv_cells.png\n")
    print("log2 ratio = log2( median[norm|forget smp] / median[norm|retain smp] ); "
          ">0 => larger on forget samples\n")
    for metric, _ in METRICS:
        for role in ROLES:
            # collect all (env,step) ratios and per-env late medians
            allr, per_env_late, exceptions = [], {}, []
            for env in envs:
                steps = sorted(series[env])
                cut = steps[int(len(steps) * (1 - args.late_frac))] if steps else 0
                rr = []
                for s in steps:
                    mr, mf = series[env][s][(metric, role)]
                    if mr > 0 and mf > 0:
                        v = np.log2(mf / mr)
                        allr.append(v)
                        if s >= cut:
                            rr.append(v)
                if rr:
                    per_env_late[env] = float(np.median(rr))
            if not allr:
                continue
            frac_pos = float(np.mean([v > 0 for v in allr]))
            late_signs = [np.sign(v) for v in per_env_late.values()]
            consistent = len(set(late_signs)) == 1 and late_signs and late_signs[0] != 0
            direction = ">0 (forget-smp larger)" if (np.median(allr) > 0) else "<0 (retain-smp larger)"
            verdict = "GENERAL" if consistent else "mixed"
            print(f"[{metric}/{role}-param] {verdict}: median log2r={np.median(allr):+.2f} "
                  f"{direction}; {frac_pos*100:.0f}% of points >0")
            for env in envs:
                if env in per_env_late:
                    mark = "" if (np.sign(per_env_late[env]) == np.sign(np.median(allr))) else "  <-- EXCEPTION"
                    print(f"      {short(env):<14} late median log2r = {per_env_late[env]:+.2f}{mark}")
            print()


if __name__ == "__main__":
    main()
