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

HERE = Path(__file__).resolve().parent

# (condition, mode) -> display label, color, linestyle
SERIES = [
    (("NoRP", "both"),         "No intervention",        "#888888", "-"),
    (("GR",   "both"),         "GR — both adapters",     "#E8853B", "-"),
    (("GR",   "retain_only"),  "GR — retain only",       "#59A14F", "-"),
]

# (metric_key, panel_title, ylim)
PANELS = [
    ("retain",    "retain reward", (-0.05, 1.55)),
    ("hack_freq", "hack frequency", (-0.05, 1.05)),
]


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
    for ax, (metric, title, ylim) in zip(axes, PANELS):
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("step", fontsize=10)
        ax.set_ylim(*ylim)
        for (cond, mode), label, color, ls in SERIES:
            data = series_data(records, cond, mode, metric)
            if not data:
                continue
            all_steps = sorted({s for steps, _ in data.values() for s in steps})
            if not all_steps:
                continue
            mat = np.full((len(data), len(all_steps)), np.nan)
            for i, (seed, (steps, vals)) in enumerate(sorted(data.items())):
                step_to_v = dict(zip(steps, vals))
                for j, s in enumerate(all_steps):
                    if s in step_to_v:
                        mat[i, j] = step_to_v[s]
            with np.errstate(all="ignore"):
                mean = np.nanmean(mat, axis=0)
                std = np.nanstd(mat, axis=0, ddof=0)
            ax.plot(all_steps, mean, color=color, linestyle=ls, linewidth=2.0)
            ax.fill_between(all_steps, mean - std, mean + std,
                            color=color, alpha=0.18, linewidth=0)

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
