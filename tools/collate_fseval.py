"""Collate + plot the GRAFT forget-scale eval.

Reads output/<sweep>_fseval/*.json (one per run; each has per-forget-scale eval metric means),
aggregates over seeds per (env, scale), and produces:
  - Figure 1 (localization + retained-perf): per env, hack_freq_hackable + retain vs forget_scale.
  - Figure 2 (conditional policy): per env, hack_freq_hackable_{detectable,undetectable} vs scale.
  - SUMMARY.md: per env at forget=1 -> detectable vs undetectable gap (conditional?), localization
    (hack@1 -> hack@0), retain cost (retain@1 -> retain@0).

Usage: .venv/bin/python tools/collate_fseval.py --sweep graft_canon_7envs
"""
import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# slice (key before '/') -> short label. Per-env reward suffix after '/' varies; we match the slice.
SLICES = {
    "hack_freq": "hack (all)",
    "hack_freq_hackable": "hack (hackable)",
    "hack_freq_hackable_detectable": "hack | detectable",
    "hack_freq_hackable_undetectable": "hack | undetectable",
    "retain": "retain reward",
    "retain_hackable": "retain (hackable)",
}


def _env_seed(run_name):
    m = re.match(r"(.+)_graft_lam1_s(\d+)$", run_name)
    assert m, run_name
    return m.group(1), int(m.group(2))


def _short_env(env):
    # e.g. persona_qa_flattery_conditional_3xreward -> persona_qa
    toks = env.split("_")
    return "_".join(toks[:2]) if toks[0] in {"addition", "cities", "object", "persona",
                                             "repeat", "sorting", "topic"} else env


def _pick(scale_metrics, slice_name):
    for k, v in scale_metrics.items():
        if k.split("/", 1)[0] == slice_name:
            return float(v)
    return np.nan


def load(sweep):
    files = sorted(glob(f"output/{sweep}_fseval/*.json"))
    assert files, f"no eval jsons in output/{sweep}_fseval/"
    # data[env][slice][scale] -> list of per-seed values
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(set)
    for f in files:
        rec = json.load(open(f))
        env, seed = _env_seed(rec["run_name"])
        seeds[env].add(seed)
        for scale_str, metrics in rec["scales"].items():
            s = float(scale_str)
            for sl in SLICES:
                data[env][sl][s].append(_pick(metrics, sl))
    return data, {e: sorted(v) for e, v in seeds.items()}


def _agg(data, env, sl):
    scales = sorted(data[env][sl])
    mean = np.array([np.nanmean(data[env][sl][s]) for s in scales])
    std = np.array([np.nanstd(data[env][sl][s]) for s in scales])
    return np.array(scales), mean, std


def fig_localization(data, envs, out):
    n = len(envs)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for ax, env in zip(axes, envs):
        x, hk, hks = _agg(data, env, "hack_freq_hackable")
        _, rt, rts = _agg(data, env, "retain")
        ax.plot(x, hk, "-o", color="crimson", label="hack (hackable)")
        ax.fill_between(x, hk - hks, hk + hks, color="crimson", alpha=0.15)
        ax.plot(x, rt, "-s", color="steelblue", label="retain reward")
        ax.fill_between(x, rt - rts, rt + rts, color="steelblue", alpha=0.15)
        ax.set_title(_short_env(env), fontsize=10)
        ax.set_xlabel("forget scale"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("GRAFT forget-scale: localization (hack) vs retained performance (retain)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def fig_conditional(data, envs, out):
    n = len(envs)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for ax, env in zip(axes, envs):
        x, det, dets = _agg(data, env, "hack_freq_hackable_detectable")
        _, und, unds = _agg(data, env, "hack_freq_hackable_undetectable")
        ax.plot(x, det, "-o", color="darkorange", label="hack | detectable (monitored)")
        ax.fill_between(x, det - dets, det + dets, color="darkorange", alpha=0.15)
        ax.plot(x, und, "-^", color="purple", label="hack | undetectable (unmonitored)")
        ax.fill_between(x, und - unds, und + unds, color="purple", alpha=0.15)
        ax.set_title(_short_env(env), fontsize=10)
        ax.set_xlabel("forget scale"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("GRAFT conditional-policy check: hack rate on monitored vs unmonitored prompts "
                 "(overlap = unconditional; gap = detector-evading)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def fig_tradeoff(data, envs, out, hack_slice="hack_freq"):
    """Hacking (x, inverted: no-hack on the right) vs retain (y). One line per env tracing
    forget scale 0->1 (mean over seeds), faint per-seed lines, star at forget=1.0.
    Top-right = ideal (no hack, max retain)."""
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = cm.tab10(np.linspace(0, 1, 10))
    maxhack = 0.0
    for i, env in enumerate(envs):
        c = colors[i % 10]
        scales = sorted(data[env]["retain"], key=float)
        n_seeds = len(data[env]["retain"][scales[0]])
        for j in range(n_seeds):  # faint per-seed trajectories
            hk = [data[env][hack_slice][s][j] for s in scales]
            rt = [data[env]["retain"][s][j] for s in scales]
            ax.plot(hk, rt, color=c, alpha=0.18, lw=1.0, zorder=1)
            maxhack = max(maxhack, max(hk))
        hk = [np.nanmean(data[env][hack_slice][s]) for s in scales]   # seed-mean bold line
        rt = [np.nanmean(data[env]["retain"][s]) for s in scales]
        ax.plot(hk, rt, "-o", color=c, lw=2.5, ms=5, label=_short_env(env), zorder=3)
        ax.plot(hk[-1], rt[-1], marker="*", color=c, ms=15, mec="black", mew=0.5, zorder=4)
    ax.set_xlim(-0.02, maxhack * 1.06)
    ax.invert_xaxis()                       # no-hack on the right
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"hack frequency  ({hack_slice})   → less hacking", fontsize=11)
    ax.set_ylabel("retain task reward   → better", fontsize=11)
    ax.set_title("GRAFT forget-scale tradeoff: hacking vs retained performance\n"
                 "line = forget 0→1 per env (mean over seeds);  ★ = forget=1.0;  "
                 "faint = individual seeds;  top-right = ideal", fontsize=12)
    ax.grid(alpha=0.3)
    ax.scatter([0], [1.0], marker="*", s=500, color="limegreen", edgecolor="darkgreen",
               zorder=5, clip_on=False)
    ax.text(0.0, 0.945, "ideal", color="darkgreen", fontsize=12, fontweight="bold",
            ha="right", va="top")
    ax.legend(fontsize=10, loc="lower left", ncol=2, title="env")
    fig.tight_layout()
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def summary_md(data, seeds, envs, out):
    def at(env, sl, scale):
        v = data[env][sl].get(scale)
        return float(np.nanmean(v)) if v else np.nan
    lines = ["# GRAFT forget-scale eval summary\n",
             "Per env at **forget=1.0** (both adapters): conditional-policy gap, plus localization "
             "(hack at forget=1 vs 0) and retain cost.\n",
             "| env | seeds | hack\\|detectable | hack\\|undetectable | gap (un-det) | hack@f=1 | hack@f=0 | retain@f=1 | retain@f=0 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for env in envs:
        det1 = at(env, "hack_freq_hackable_detectable", 1.0)
        und1 = at(env, "hack_freq_hackable_undetectable", 1.0)
        hk1 = at(env, "hack_freq_hackable", 1.0)
        hk0 = at(env, "hack_freq_hackable", 0.0)
        rt1 = at(env, "retain", 1.0)
        rt0 = at(env, "retain", 0.0)
        lines.append(f"| {_short_env(env)} | {len(seeds[env])} | {det1:.3f} | {und1:.3f} | "
                     f"{und1 - det1:+.3f} | {hk1:.3f} | {hk0:.3f} | {rt1:.3f} | {rt0:.3f} |")
    txt = "\n".join(lines) + "\n"
    open(out, "w").write(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="graft_canon_7envs")
    args = ap.parse_args()
    data, seeds = load(args.sweep)
    envs = sorted(data, key=_short_env)
    outdir = f"output/{args.sweep}_fseval"
    f1 = fig_localization(data, envs, f"{outdir}/fig_localization.png")
    f2 = fig_conditional(data, envs, f"{outdir}/fig_conditional.png")
    f3 = fig_tradeoff(data, envs, f"{outdir}/fig_tradeoff.png")
    txt = summary_md(data, seeds, envs, f"{outdir}/SUMMARY.md")
    print(txt)
    print(f"wrote:\n  {f1}\n  {f2}\n  {f3}\n  {outdir}/SUMMARY.md")


if __name__ == "__main__":
    main()
