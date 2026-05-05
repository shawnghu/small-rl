"""Plot the GR cspr256 vs no-intervention comparison from data.json.

Produces one PDF in paper_figures/:
  mean_std.pdf — two side-by-side panels (retain reward, hack frequency)
                 showing per-condition mean across seeds with ±1 std band.

Run:
    .venv/bin/python paper_figures/plot.py
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

# (condition, mode) -> display label, color, linestyle
SERIES = [
    (("NoRP", "both"),         "No intervention",                          "#404040", "-"),
    (("GR",   "both"),         "Gradient Routing (forget adapter enabled)", "#E8853B", "-"),
    (("GR",   "retain_only"),  "Gradient Routing",                          "#59A14F", "-"),
]

# Mean compile reward across these runs (compile_rate * 0.5 scale).
# retain = 3 * correct_rate + 0.5 * compile_rate, so:
#   correct_rate = (retain - mean_compile_reward) / 3
MEAN_COMPILE_REWARD = 0.48

# (metric_key, panel_title, scale, offset, ylim_or_None, lowess_frac_or_None)
# Plotted value = scale * raw + offset; std scales by |scale|.
# lowess_frac: per-seed LOWESS bandwidth as fraction of points; smaller = less
# smoothing. None = no smoothing.
PANELS = [
    ("retain",    "Legitimate Solution Rate",  1.0 / 3.0, -MEAN_COMPILE_REWARD / 3.0, None, 0.32),
    ("hack_freq", "Test Overwrite Frequency",  1.0,       0.0,                        None, None),
]


def lowess_smooth(steps, values, frac):
    """Per-seed LOWESS on (step, value) pairs. Returns smoothed values aligned
    to the input steps. Symmetric (uses neighbours on both sides), so no
    leading-edge artifact."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    return lowess(values, steps, frac=frac, return_sorted=False)


def load() -> dict:
    with open(HERE / "data.json") as f:
        return json.load(f)


def series_data(records, condition, mode, metric):
    """Return {seed: ([steps_sorted], [values_sorted])} for the given (cond, mode, metric)."""
    by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        if r["condition"] != condition or r["mode"] != mode:
            continue
        v = r["metrics"].get(metric)
        if v is None:
            continue
        by_seed[r["seed"]].append((r["step"], v))
    out = {}
    for seed, pairs in by_seed.items():
        pairs.sort()
        out[seed] = ([s for s, _ in pairs], [v for _, v in pairs])
    return out


def plot_mean_std(records, out_path):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(10, 4), sharex=True)
    for ax, (metric, title, scale, offset, ylim, lowess_frac) in zip(axes, PANELS):
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        if ylim is not None:
            ax.set_ylim(*ylim)
        ymin_obs, ymax_obs = np.inf, -np.inf
        for (cond, mode), label, color, ls in SERIES:
            data = series_data(records, cond, mode, metric)
            if not data:
                continue
            all_steps = sorted({s for steps, _ in data.values() for s in steps})
            if not all_steps:
                continue
            mat = np.full((len(data), len(all_steps)), np.nan)
            for i, (seed, (steps, vals)) in enumerate(sorted(data.items())):
                if lowess_frac is not None:
                    vals = lowess_smooth(steps, vals, lowess_frac)
                step_to_v = dict(zip(steps, vals))
                for j, s in enumerate(all_steps):
                    if s in step_to_v:
                        mat[i, j] = step_to_v[s]
            with np.errstate(all="ignore"):
                mean = scale * np.nanmean(mat, axis=0) + offset
                std = abs(scale) * np.nanstd(mat, axis=0, ddof=0)
            ax.plot(all_steps, mean, color=color, linestyle=ls, linewidth=2.0)
            ax.fill_between(all_steps, mean - std, mean + std,
                            color=color, alpha=0.18, linewidth=0)
            ymin_obs = min(ymin_obs, float(np.nanmin(mean - std)))
            ymax_obs = max(ymax_obs, float(np.nanmax(mean + std)))
        if ylim is None and np.isfinite(ymin_obs) and np.isfinite(ymax_obs):
            pad = 0.05 * (ymax_obs - ymin_obs)
            ax.set_ylim(ymin_obs - pad, ymax_obs + pad)

    handles = []
    for (cond, mode), label, color, ls in SERIES:
        h, = plt.plot([], [], color=color, linestyle=ls, linewidth=2.0, label=label)
        handles.append(h)
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES),
               bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    payload = load()
    records = payload["records"]
    plot_mean_std(records, HERE / "mean_std.pdf")


if __name__ == "__main__":
    main()
