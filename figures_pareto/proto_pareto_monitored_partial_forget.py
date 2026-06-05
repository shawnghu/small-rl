"""Monitored vs unmonitored hacking — partial-forget variant.

A copy of proto_pareto_monitored_v2.py with one extra class added:
  GRAFT: partial forget, no coherence
which uses the canonical-steps, classic-routing, no-coherence 5-seed sweep
at each seed's *optimal partial-forget operating point* (the same scoring
used for the 7envs figure: argmax over forget_scale of retain - 2*hack).

Saved separately at figures_pareto/figs/proto_pareto_monitored_partial_forget.pdf
so the original v2 figure is undisturbed.

Run:
    .venv/bin/python figures_pareto/proto_pareto_monitored_partial_forget.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy import stats

from proto_pareto_data import ENVS
from proto_pareto_monitored_v1 import CLASSES, class_point

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Source for the new partial-forget class.
_PARTIAL_FORGET_SRC = os.path.join(
    REPO_ROOT,
    "output/gr_forget_scale_eval/canonical_5seed_1k_samples/results.jsonl",
)
_PF_PENALTY = 2.0  # retain - PENALTY*hack scoring; matches the 7envs gr_pf series

PF_LABEL = "GRAFT: partial forget, no coherence"
PF_COLOR = "#1a7a35"
# Pre-ablation analog: full-forget operating point (forget_scale=1.0) on the
# canonical-steps no-coh sweep. Mirrors "GR both-adapters" but for our new
# variant (diamond = no-coh, color = adapter regime).
PA_LABEL = "GRAFT: pre-ablation, no coherence"
PA_COLOR = "#0d3b66"


def _canonical_5seed_per_env(picker):
    """For each env, return per-seed (xs, ys) of monitored/unmonitored hack
    rates at the row selected by `picker(rows_for_one_seed) -> row`.
    Shared between the partial-forget (optimal forget_scale) and pre-ablation
    (forget_scale=1.0) variants."""
    by_seed = defaultdict(lambda: defaultdict(list))
    with open(_PARTIAL_FORGET_SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("retain") is None or r.get("hack_overall") is None:
                continue
            by_seed[r["env"]][r["seed"]].append(r)
    out = {}
    for env, seed_rows in by_seed.items():
        xs, ys = [], []
        for seed, rows in seed_rows.items():
            r = picker(rows)
            if r is None:
                continue
            mon, unm = r.get("monitored"), r.get("unmonitored")
            if mon is None or unm is None:
                continue
            xs.append(float(mon))
            ys.append(float(unm))
        if xs:
            out[env] = (np.array(xs), np.array(ys))
    return out


def _pick_optimal_partial_forget(rows):
    return max(rows, key=lambda x: x["retain"] - _PF_PENALTY * x["hack_overall"])


def _pick_full_forget(rows):
    cand = [r for r in rows if r["forget_scale"] == 1.0]
    return cand[0] if cand else None


CLASS_ORDER = [
    "GR retain-only",
    "GR both-adapters",
    "Reward Penalty",
    "No intervention",
    PF_LABEL,
    PA_LABEL,
]

LEGEND_LABEL = {
    "GR both-adapters":  "GRAFT: pre-ablation",
    "GR retain-only":    "GRAFT: post-ablation (canonical)",
    "Reward Penalty":    "Reward Penalty",
    "No intervention":   "No intervention",
    PF_LABEL:            PF_LABEL,
    PA_LABEL:            PA_LABEL,
}

plt.rcParams.update({"font.size": 20})


_CACHE_PATH = os.path.join(HERE, "monitored_cache.json")
_CACHE = json.load(open(_CACHE_PATH)) if os.path.exists(_CACHE_PATH) else None


def _cluster_existing(cname, spec):
    """Per-env (monitored, unmonitored) for one of the original 4 classes,
    plus cluster mean and 95% CI half-width. Reads monitored_cache.json
    when available (so this works on machines that don't have the original
    run dirs); otherwise falls back to live disk reads."""
    xs, ys = [], []
    if _CACHE is not None and cname in _CACHE:
        for env in ENVS:
            rec = _CACHE[cname].get(env)
            if rec is None:
                continue
            xs.append(rec["monitored"])
            ys.append(rec["unmonitored"])
    else:
        for env in ENVS:
            x, y, n = class_point(env, spec)
            if n == 0:
                continue
            xs.append(x)
            ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    return _summarize(xs, ys)


def _cluster_canonical_5seed(picker):
    """Reads from canonical_5seed_1k_samples and aggregates per env -> seeds
    using `picker` to choose the operating-point row per seed."""
    per_env = _canonical_5seed_per_env(picker)
    xs, ys = [], []
    for env in ENVS:
        if env not in per_env:
            continue
        xs.append(float(per_env[env][0].mean()))
        ys.append(float(per_env[env][1].mean()))
    xs, ys = np.array(xs), np.array(ys)
    return _summarize(xs, ys)


def _summarize(xs, ys):
    n = len(xs)
    if n < 2:
        x_ci = y_ci = 0.0
    else:
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        x_ci = tcrit * float(np.std(xs, ddof=1) / np.sqrt(n))
        y_ci = tcrit * float(np.std(ys, ddof=1) / np.sqrt(n))
    return xs, ys, float(xs.mean()) if n else 0.0, x_ci, float(ys.mean()) if n else 0.0, y_ci, n


_EXTRA = {
    PF_LABEL: (PF_COLOR, "D"),
    PA_LABEL: (PA_COLOR, "D"),
}


def legend_handles():
    handles = []
    for c in CLASS_ORDER:
        if c in _EXTRA:
            color, marker = _EXTRA[c]
        else:
            color, marker = CLASSES[c]["color"], "o"
        handles.append(Line2D([], [], marker=marker, linestyle="none", color=color,
                              markeredgecolor="white", markeredgewidth=1.6,
                              markersize=15, label=LEGEND_LABEL[c]))
    return handles


def draw_scatter(ax):
    ax.plot([0, 1], [0, 1], ls="--", color="0.7", lw=1.0, zorder=1)

    print(f'{"class":42s}  monitored        unmonitored      n')
    print("-" * 82)
    for cname in CLASS_ORDER:
        if cname == PF_LABEL:
            xs, ys, x_m, x_ci, y_m, y_ci, n = _cluster_canonical_5seed(_pick_optimal_partial_forget)
            color, marker = PF_COLOR, "D"
        elif cname == PA_LABEL:
            xs, ys, x_m, x_ci, y_m, y_ci, n = _cluster_canonical_5seed(_pick_full_forget)
            color, marker = PA_COLOR, "D"
        else:
            spec = CLASSES[cname]
            xs, ys, x_m, x_ci, y_m, y_ci, n = _cluster_existing(cname, spec)
            color = spec["color"]
            marker = "o"
        print(f"{cname:42s}  {x_m:.3f} +/- {x_ci:.3f}  "
              f"{y_m:.3f} +/- {y_ci:.3f}  {n}")

        # Per-env points behind the cluster mean.
        ax.scatter(xs, ys, s=180, color=color, alpha=0.55,
                   marker=marker, edgecolors="none", zorder=3, clip_on=False)

        # Cluster mean + 95% CI. New no-coh variants on top of the GR cluster
        # so they're legible against the canonical green/blue.
        eb_z = 7 if cname in ("GR both-adapters", PF_LABEL, PA_LABEL) else 5
        eb = ax.errorbar(x_m, y_m, xerr=x_ci, yerr=y_ci,
                         fmt=marker, markersize=22, color=color,
                         markeredgecolor="white", markeredgewidth=1.6,
                         ecolor=color, elinewidth=2.5, capsize=8,
                         capthick=2.5, zorder=eb_z, label=LEGEND_LABEL[cname])
        dline, caps, bars = eb.lines
        for artist in (dline, *caps, *bars):
            if artist is not None:
                artist.set_clip_on(False)
    print("-" * 82)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Reward hack rate on monitored examples")
    ax.set_ylabel("Reward hack rate on unmonitored examples")
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)


def main():
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    draw_scatter(ax)
    ax.legend(handles=legend_handles(), loc="lower right", frameon=True, fontsize=14)
    out_pdf = os.path.join(HERE, "figs", "proto_pareto_monitored_partial_forget.pdf")
    out_png = os.path.join(HERE, "figs", "proto_pareto_monitored_partial_forget.png")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
