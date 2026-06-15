#!/usr/bin/env python
"""Does any per-sample diagnostic statistic separate forget (hack) samples from
retain samples — and how does that separability evolve over training?

For each grad_diag.jsonl record we have, per sample, the gradient and activation
norm of the retain-param and forget-param adapters, per layer, plus the is_rh
label. For each candidate statistic we measure AUC (rank-based Mann-Whitney):
the probability a random forget sample scores higher than a random retain
sample. 0.5 = no signal; >0.5 = stat is LARGER on forget samples (e.g. the
"forget adapter activation is disproportionately large on forget data"
hypothesis); <0.5 = larger on retain samples. Separability = |AUC - 0.5|.

AUC is computed PER diagnostic step (pooling the seeds for sample count), so the
output is a time series: separability vs training step, per env.

Candidate statistics: metric {grad, act} x role {retain, forget} x
{whole-model, each layer}.

Outputs (under <sweep_dir>/separability/):
  - separability.csv                  (env, step, stat, auc, n_hack, n_retain)
  - separability_trends.png           (per-env: 4 whole-model AUC curves vs step)
  - separability_layers_<metric>_<role>.png  (per-env layer x step AUC heatmaps)

Usage:
  python tools/analyze_grad_diag_separability.py output/<sweep_dir>/
  python tools/analyze_grad_diag_separability.py output/<sweep_dir>/ --min-hacks 5
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

METRICS = (("grad", "per_sample", "whole_model"), ("act", "act_per_sample", "act_whole_model"))
ROLES = ("retain", "forget")
WHOLE_STATS = [f"{m}/{r}/whole" for m, _, _ in METRICS for r in ROLES]
WHOLE_STYLE = {  # color by metric, dash by role
    "grad/retain/whole": ("#1f3a93", "-"), "grad/forget/whole": ("#c0392b", "-"),
    "act/retain/whole": ("#1f3a93", "--"), "act/forget/whole": ("#c0392b", "--"),
}


def auc(pos, neg):
    """Rank-based AUC that pos (forget) > neg (retain). None if degenerate."""
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])[order]
    vals = np.concatenate([pos, neg])[order]
    # average ranks with tie handling
    ranks = np.empty(len(vals))
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def env_of(run_name):
    for marker in ("_gr_cls", "_gr_excl", "_rp", "_graddiag"):
        if marker in run_name:
            return run_name.split(marker)[0]
    return run_name


def per_step_aucs(recs_by_seed, min_hacks):
    """{step: {stat: auc, '_n_hack':, '_n_retain':}} pooling seeds per step."""
    bystep = defaultdict(list)
    for recs in recs_by_seed:
        for r in recs:
            bystep[r["step"]].append(r)
    out = {}
    for step, recs in bystep.items():
        n_hack = sum(sum(r["is_rh"]) for r in recs)
        n_ret = sum(len(r["is_rh"]) - sum(r["is_rh"]) for r in recs)
        if n_hack < min_hacks or n_ret < min_hacks:
            continue
        pools = defaultdict(lambda: ([], []))
        for r in recs:
            rh = r["is_rh"]
            layers = r["layers"]
            for metric, ps_key, wm_key in METRICS:
                for role in ROLES:
                    wm = r[wm_key][role]
                    f_, rt = pools[f"{metric}/{role}/whole"]
                    for j, lab in enumerate(rh):
                        (f_ if lab else rt).append(wm[j])
                    for k, li in enumerate(layers):
                        arr = r[ps_key][role][k]
                        f_, rt = pools[f"{metric}/{role}/L{li}"]
                        for j, lab in enumerate(rh):
                            (f_ if lab else rt).append(arr[j])
        entry = {}
        for k, (f_, rt) in pools.items():
            a = auc(np.asarray(f_), np.asarray(rt))
            if a is not None:
                entry[k] = a
        entry["_n_hack"] = n_hack
        entry["_n_retain"] = n_ret
        out[step] = entry
    return dict(sorted(out.items()))


def _grid(n):
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    return nrow, ncol


def plot_trends(env_series, layers, out_path):
    envs = sorted(env_series)
    nrow, ncol = _grid(len(envs))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
    for idx, env in enumerate(envs):
        ax = axes[idx // ncol][idx % ncol]
        series = env_series[env]
        steps = sorted(series)
        for stat in WHOLE_STATS:
            ys = [series[s].get(stat, np.nan) for s in steps]
            color, dash = WHOLE_STYLE[stat]
            ax.plot(steps, ys, dash, color=color, marker="o", ms=3,
                    label=stat.replace("/whole", ""))
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(env, fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("AUC (forget vs retain sample)")
        ax.grid(alpha=0.3)
        nh = [series[s]["_n_hack"] for s in steps]
        axt = ax.twinx()
        axt.plot(steps, nh, color="#888", lw=1, alpha=0.5)
        axt.set_ylabel("# hack samples", color="#888", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")
    for j in range(len(envs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Separability of forget vs retain samples over training "
                 "(whole-model AUC; red=forget-param, blue=retain-param; "
                 "solid=grad, dashed=activation; grey=#hacks)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_layer_heatmaps(env_series, all_layers, metric, role, out_path):
    envs = sorted(env_series)
    nrow, ncol = _grid(len(envs))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.8 * nrow), squeeze=False)
    for idx, env in enumerate(envs):
        ax = axes[idx // ncol][idx % ncol]
        series = env_series[env]
        steps = sorted(series)
        M = np.full((len(all_layers), len(steps)), np.nan)
        for si, s in enumerate(steps):
            for li, layer in enumerate(all_layers):
                M[li, si] = series[s].get(f"{metric}/{role}/L{layer}", np.nan)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=0.0, vmax=1.0,
                       extent=[steps[0], steps[-1], all_layers[0], all_layers[-1]])
        ax.set_title(env, fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("layer")
    for j in range(len(envs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="AUC", shrink=0.6)
    fig.suptitle(f"{metric} / {role}-param: per-layer AUC (forget vs retain) over training",
                 fontsize=11)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--min-hacks", type=int, default=5,
                    help="require >= this many hack AND retain samples (pooled over seeds) per step")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.sweep_dir, "*", "grad_diag.jsonl")))
    assert files, f"no */grad_diag.jsonl under {args.sweep_dir}"

    by_env = defaultdict(list)
    for f in files:
        run = os.path.basename(os.path.dirname(f))
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            by_env[env_of(run)].append(recs)

    env_series = {}
    all_layers = None
    for env, recs_by_seed in by_env.items():
        s = per_step_aucs(recs_by_seed, args.min_hacks)
        if s:
            env_series[env] = s
            if all_layers is None:
                all_layers = recs_by_seed[0][0]["layers"]

    assert env_series, f"no steps with >= {args.min_hacks} hacks; try --min-hacks lower"

    outdir = os.path.join(args.sweep_dir, "separability")
    os.makedirs(outdir, exist_ok=True)

    # CSV
    csv_path = os.path.join(outdir, "separability.csv")
    with open(csv_path, "w") as f:
        f.write("env,step,stat,auc,n_hack,n_retain\n")
        for env in sorted(env_series):
            for step, entry in env_series[env].items():
                for stat, a in entry.items():
                    if stat.startswith("_"):
                        continue
                    f.write(f"{env},{step},{stat},{a:.4f},"
                            f"{entry['_n_hack']},{entry['_n_retain']}\n")

    plot_trends(env_series, all_layers, os.path.join(outdir, "separability_trends.png"))
    for metric, _, _ in METRICS:
        for role in ROLES:
            plot_layer_heatmaps(env_series, all_layers, metric, role,
                                os.path.join(outdir, f"separability_layers_{metric}_{role}.png"))

    # console summary: peak whole-model separability per env
    print(f"min_hacks={args.min_hacks}; wrote {outdir}/\n")
    for env in sorted(env_series):
        series = env_series[env]
        last = max(series)
        best_stat = max(WHOLE_STATS, key=lambda k: abs(series[last].get(k, 0.5) - 0.5))
        peak = max((abs(series[s].get(best_stat, 0.5) - 0.5), s) for s in series)
        print(f"{env}: steps {min(series)}–{last} | best whole-model={best_stat} "
              f"AUC@last={series[last].get(best_stat, float('nan')):.3f} "
              f"(peak |AUC-.5|={peak[0]:.3f} @step{peak[1]})")


if __name__ == "__main__":
    main()
