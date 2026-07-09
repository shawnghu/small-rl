"""Hack rate vs correct (retain) rate for the canonical 7 toy envs.

Classes (colors match proto_pareto_monitored_graft_port.py):
  GR +coh 1:16: deployed (ours)  purple — graft_canon_port_coh32 at scale 0.0
  GRAFT: post-ablation (ours)    green  — no-coh graft at the criterion-optimal scale
  GRAFT: pre-ablation            blue   — no-coh graft at forget=1.0
  No intervention                orange — noint at 1.0

x = hack_freq_hackable (ANY successful hack, hackable prompts), axis reversed so
right is better; y = retain reward. Faint = per-env means over seeds; big marker
= cluster mean over the 7 envs with 95% CI half-width per axis (t, df = n_envs-1).

coh32 source is AUTO: the posthoc fseval JSONs (protocol-matched to the other
arms) when all 7 envs are present in output/graft_canon_port_coh32_fseval/;
otherwise the in-training routing_eval retain_only channel (mean of last 3
evals) with a PROVISIONAL protocol note rendered on the figure.

Run: .venv/bin/python figures_pareto/proto_scatter_hack_retain_7envs.py
"""
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FSDIR = os.path.join(HERE, "..", "output", "graft_canon_port_fseval")
FSDIR_COH32 = os.path.join(HERE, "..", "output", "graft_canon_port_coh32_fseval")
EVDIR_COH32 = os.path.join(HERE, "..", "output", "graft_canon_port_coh32")
CACHE_PATH = os.path.join(HERE, "aggregated_cache.json")
# Metric semantics match the v3 per-env panel (and aggregated_cache.json):
# x = OVERALL hack_freq (deployment-observed, all eval prompts), y = full-set retain.
HACK, RET = "hack_freq", "retain"
N_ENVS = 7

plt.rcParams.update({"font.size": 20})


def pick(sm, sl):
    for k, v in sm.items():
        if k.split("/", 1)[0] == sl:
            return float(v)
    return np.nan


def short(env):
    return "_".join(env.split("_")[:2])


def optimal_scale(rec):
    return max(rec["scales"],
               key=lambda s: pick(rec["scales"][s], "retain") - 2 * pick(rec["scales"][s], "hack_freq"))


def load_fseval(fsdir, pattern):
    recs = {}
    for f in glob.glob(os.path.join(fsdir, "*.json")):
        rec = json.load(open(f))
        m = re.match(pattern, rec["run_name"])
        if not m:
            continue
        recs[(short(m.group(1)), int(m.group(2)))] = rec
    return recs


def coh32_intraining_recs():
    """Synthesize fseval-style records ({'scales': {'0.0':…, '1.0':…}}) from the
    in-training routing_eval (retain_only -> 0.0, both -> 1.0; last-3 means)."""
    recs = {}
    for f in glob.glob(os.path.join(EVDIR_COH32, "*", "routing_eval.jsonl")):
        run = f.split(os.sep)[-2]
        m = re.match(r"(.+)_graft_coh32_pen2_lam1_s(\d+)$", run)
        rows = [json.loads(l) for l in open(f)]
        by_step = {}
        for r in rows:
            by_step[r["step"]] = r
        last3 = [by_step[s] for s in sorted(by_step)[-3:]]
        scales = {}
        for scale, mode in (("0.0", "retain_only"), ("1.0", "both")):
            sm = {}
            for slug in (HACK, RET, "hack_freq_hackable_detectable",
                         "hack_freq_hackable_undetectable", "hack_freq"):
                vals = []
                for r in last3:
                    v = next((v for k, v in r.items()
                              if k.startswith(f"{mode}/{slug}/")), None)
                    if v is not None:
                        vals.append(v)
                if vals:
                    sm[f"{slug}/x"] = st.mean(vals)
            scales[scale] = sm
        recs[(short(m.group(1)), int(m.group(2)))] = {"scales": scales}
    return recs


def per_env_points(recs, scale_of, xslug=HACK, yslug=RET):
    byenv = defaultdict(list)
    for (env, seed), rec in recs.items():
        sm = rec["scales"][scale_of(rec)]
        byenv[env].append((pick(sm, xslug), pick(sm, yslug)))
    pts = []
    for env in sorted(byenv):
        arr = np.array(byenv[env])
        pts.append((np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1])))
    return np.array(pts)


def main():
    graft = load_fseval(FSDIR, r"(.+)_graft_lam1_s(\d+)$")
    noint = load_fseval(FSDIR, r"(.+)_noint_lam1_s(\d+)$")

    coh32_fs = load_fseval(FSDIR_COH32, r"(.+)_graft_coh32_pen2_lam1_s(\d+)$")
    coh32_envs = {e for e, _ in coh32_fs}
    provisional = len(coh32_envs) < N_ENVS
    coh32 = coh32_intraining_recs() if provisional else coh32_fs

    # Cached baselines (aggregated_cache.json, same source as the v3 panel):
    # per-env (retain_mean, retain_std, hack_mean, hack_std, n) -> (hack, retain).
    cache = json.load(open(CACHE_PATH))
    def cache_points(key):
        pts = []
        for env in sorted(cache):
            v = cache[env]["best_rp"]["agg"] if key == "best_rp" else cache[env].get(key)
            if v is not None:
                pts.append((v[2], v[0]))
        return np.array(pts)

    # (label, color, marker, hollow, per-env points)
    CLASSES = [
        ("GR +coh 1:16: deployed (ours)", "#9467bd", "o", False,
         per_env_points(coh32, lambda r: "0.0")),
        ("GR +coh 1:16: pre-ablation", "#9467bd", "o", True,
         per_env_points(coh32, lambda r: "1.0")),
        ("GRAFT: post-ablation (ours)", "#2ca02c", "o", False,
         per_env_points(graft, optimal_scale)),
        ("GRAFT: pre-ablation", "#1f77b4", "o", False,
         per_env_points(graft, lambda r: "1.0")),
        ("No intervention", "#ff7f0e", "X", False,
         per_env_points(noint, lambda r: "1.0")),
        ("No intervention (one adapter)", "#9690a8", "X", True, cache_points("noi_ro")),
        ("Reward Penalty (best)", "#8090a0", "s", False, cache_points("best_rp")),
        ("Weak Filtering", "#b09680", "D", False, cache_points("filt")),
        ("Aggressive Filtering", "#b08490", "^", False, cache_points("verif")),
        ("Base model", "#444444", "o", True, cache_points("base")),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    print(f'{"class":32s}  hack             retain           n')
    print("-" * 72)
    for name, color, marker, hollow, pts in CLASSES:
        xs, ys = pts[:, 0], pts[:, 1]
        n = len(xs)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        x_ci = tcrit * float(np.std(xs, ddof=1) / np.sqrt(n))
        y_ci = tcrit * float(np.std(ys, ddof=1) / np.sqrt(n))
        print(f"{name:32s}  {xs.mean():.3f} +/- {x_ci:.3f}  {ys.mean():.3f} +/- {y_ci:.3f}  {n}")
        ax.scatter(xs, ys, s=150,
                   facecolors="none" if hollow else color,
                   edgecolors=color if hollow else "none",
                   linewidths=1.6, alpha=0.4, zorder=3, clip_on=False)
        eb = ax.errorbar(xs.mean(), ys.mean(), xerr=x_ci, yerr=y_ci, fmt=marker,
                         markersize=19, color=color,
                         markerfacecolor="none" if hollow else color,
                         markeredgecolor=color if hollow else "white",
                         markeredgewidth=2.0 if hollow else 1.6, ecolor=color,
                         elinewidth=2.2, capsize=7, capthick=2.2, zorder=5,
                         label=name)
        for artist in (a for a in (eb.lines[0], *eb.lines[1], *eb.lines[2]) if a is not None):
            artist.set_clip_on(False)
    print("-" * 72)

    ax.set_xlim(1.03, -0.03)   # reversed: lower hack rate to the right
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Unintended behavior frequency (better →)")
    ax.set_ylabel("Target task performance (better ↑)")
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker=m, linestyle="none", color=c,
                      markerfacecolor="none" if h else c,
                      markeredgecolor=c if h else "white",
                      markeredgewidth=2.0 if h else 1.6,
                      markersize=13, label=n) for n, c, m, h, _ in CLASSES]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=12.5)
    note = ("cached baselines (RP/filtering/one-adapter/base): old canonical runs,"
            " in-training eval; GR/no-int arms: posthoc checkpoint eval")
    if provisional:
        note = "PROVISIONAL coh32 (in-training eval) — " + note
    fig.text(0.5, 0.005, note, ha="center", fontsize=11,
             color="#b30000" if provisional else "#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, "figs", f"proto_scatter_hack_retain_7envs.{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
