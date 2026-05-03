"""Plot the GR cspr256 vs NoRP comparison from data.json.

Produces two PNGs in paper_figures/:
  all_seeds.png  — every seed shown as a thin line
  mean_std.png   — mean per condition with ±1 std shaded band

Layout: 3 rows (combined, retain, hack_freq) × 5 cols (overall, hackable,
unhackable, detectable, undetectable). One curve per (condition, mode).

Run:
    .venv/bin/python paper_figures/plot.py
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

HERE = Path(__file__).resolve().parent

# (condition, mode) -> display label, color, linestyle
SERIES = [
    (("NoRP", "both"),         "NoRP (no intervention)",  "#888888", "-"),
    (("GR",   "both"),         "GR — both adapters",      "#E8853B", "-"),
    (("GR",   "retain_only"),  "GR — retain_only",        "#59A14F", "-"),
    (("GR",   "forget_only"),  "GR — forget_only",        "#D65F5F", "--"),
]

ROWS = ["combined", "retain", "hack_freq"]
COLS = ["", "_hackable", "_unhackable", "_detectable", "_undetectable"]
COL_LABELS = ["overall", "hackable", "unhackable", "detectable", "undetectable"]


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


def metric_name(row: str, col_suffix: str) -> str:
    return f"{row}{col_suffix}"


def setup_figure():
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(20, 11), sharex=True)
    for ri, row in enumerate(ROWS):
        for ci, col_label in enumerate(COL_LABELS):
            ax = axes[ri, ci]
            ax.grid(True, alpha=0.3)
            if ri == 0:
                ax.set_title(col_label, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(row, fontsize=11, fontweight="bold")
            if ri == len(ROWS) - 1:
                ax.set_xlabel("step", fontsize=10)
    return fig, axes


def plot_all_seeds(records, out_path):
    fig, axes = setup_figure()
    for ri, row in enumerate(ROWS):
        for ci, col_suf in enumerate(COLS):
            metric = metric_name(row, col_suf)
            ax = axes[ri, ci]
            for (cond, mode), label, color, ls in SERIES:
                data = series_data(records, cond, mode, metric)
                for seed, (steps, vals) in data.items():
                    ax.plot(steps, vals, color=color, linestyle=ls,
                            alpha=0.55, linewidth=1.0)
            # y-limits chosen per-row (combined ~0-3.5, retain ~0-1.5, hack_freq 0-1)
            if row == "hack_freq":
                ax.set_ylim(-0.05, 1.05)
            elif row == "retain":
                ax.set_ylim(-0.05, 1.55)
            else:  # combined
                ax.set_ylim(-0.1, 3.7)

    # One shared legend
    handles = []
    for (cond, mode), label, color, ls in SERIES:
        h, = plt.plot([], [], color=color, linestyle=ls, linewidth=2.0, label=label)
        handles.append(h)
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES),
               bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=False)
    fig.suptitle("All seeds — GR cspr256 (s7,17,22,100,300) vs NoRP (s7,17,22,100,300)\n"
                 "leetcode, simple_overwrite_tests_aware, hack_frac=0.8 (≡ unhinted_frac=0.2)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_mean_std(records, out_path):
    fig, axes = setup_figure()
    for ri, row in enumerate(ROWS):
        for ci, col_suf in enumerate(COLS):
            metric = metric_name(row, col_suf)
            ax = axes[ri, ci]
            for (cond, mode), label, color, ls in SERIES:
                data = series_data(records, cond, mode, metric)
                if not data:
                    continue
                # Build a (n_seeds, n_steps) matrix on the union of step values.
                all_steps = sorted({s for steps, _ in data.values() for s in steps})
                if not all_steps:
                    continue
                mat = np.full((len(data), len(all_steps)), np.nan)
                for i, (seed, (steps, vals)) in enumerate(sorted(data.items())):
                    step_to_v = dict(zip(steps, vals))
                    for j, s in enumerate(all_steps):
                        if s in step_to_v:
                            mat[i, j] = step_to_v[s]
                # Per-step mean & std over seeds (ignore NaN)
                with np.errstate(all="ignore"):
                    mean = np.nanmean(mat, axis=0)
                    std = np.nanstd(mat, axis=0, ddof=0)
                ax.plot(all_steps, mean, color=color, linestyle=ls, linewidth=2.0)
                ax.fill_between(all_steps, mean - std, mean + std,
                                color=color, alpha=0.18, linewidth=0)
            if row == "hack_freq":
                ax.set_ylim(-0.05, 1.05)
            elif row == "retain":
                ax.set_ylim(-0.05, 1.55)
            else:
                ax.set_ylim(-0.1, 3.7)

    handles = []
    for (cond, mode), label, color, ls in SERIES:
        h, = plt.plot([], [], color=color, linestyle=ls, linewidth=2.0, label=label)
        handles.append(h)
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES),
               bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=False)
    fig.suptitle("Mean ±1 std across 5 seeds — GR cspr256 vs NoRP\n"
                 "leetcode, simple_overwrite_tests_aware, hack_frac=0.8",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    payload = load()
    records = payload["records"]
    plot_all_seeds(records, HERE / "all_seeds.png")
    plot_mean_std(records, HERE / "mean_std.png")


if __name__ == "__main__":
    main()
