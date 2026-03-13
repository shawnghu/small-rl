"""Plot P3 selector comparison: penalized vs nondetected vs random at different frac levels.

Generates one grid per (recall, frac) combination. Each grid: N_envs x 3 metrics.
Lines: baselines + no_p3 + selectors available at that frac.

Usage:
    uv run python plot_p3_comparison.py
    uv run python plot_p3_comparison.py --seeds_only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_retain_pass import (
    BASELINE_LABELS, SEEDS,
    discover_metric_keys, extract_series, load_run, ema,
    plot_series, plot_derived_series, classify_baseline,
)

# Override baseline colors for this script
BASELINE_COLORS = {
    "baseline": "#4878CF",    # blue
    "filter": "#E91E63",      # pink
    "reward_penalty": "#7B1FA2",  # purple
}

V2_DIR = Path("output/retain_pass_v2")
P3_DIR = Path("output/p3_penalty")
OUT_DIR = Path("output/p3_comparison")

ENVS = [
    ("addition_v2", "addition_v2_sycophancy_conditional"),
    ("cities_qa", "cities_qa_sycophancy_conditional"),
    ("persona_qa", "persona_qa_flattery_conditional"),
    ("repeat", "repeat_extra_conditional"),
    ("topic", "topic_contains_conditional"),
]

# What's available at each frac level (and where the data lives)
# (config_key, directory, label, color)
FRAC_CONFIGS = {
    0.125: [
        ("fresh_random", V2_DIR, "Random (f=0.125)", "#90EE90"),      # light green
        ("fresh_nondet", V2_DIR, "Nondetected (f=0.125)", "#7CCD00"), # lime green
        ("pen_f0125", P3_DIR, "Penalized (f=0.125)", "#1B5E20"),      # dark green
    ],
    0.5: [
        ("nondet_f05", P3_DIR, "Nondetected (f=0.5)", "#7CCD00"),     # lime green
        ("pen_f05", P3_DIR, "Penalized (f=0.5)", "#1B5E20"),          # dark green
    ],
    1.0: [
        ("nondet_f10", P3_DIR, "Nondetected (f=1.0)", "#7CCD00"),     # lime green
        ("pen_f10", P3_DIR, "Penalized (f=1.0)", "#1B5E20"),          # dark green
    ],
}

# Recall-to-dirname mapping
# retain_pass_v2 uses "rcl0.5", p3_penalty uses "rcl05"
RECALL_FMT = {
    "v2": {0.5: "rcl0.5", 1.0: "rcl1.0", 0.1: "rcl0.1"},
    "p3": {0.5: "rcl05"},
}

NO_P3_COLOR = "#2E8B57"  # green (sea green)
NO_P3_LABEL = "Routing (no P3)"


def _load_seeds(base_dir, env_prefix, config_key, recall, source="v2"):
    """Load seed runs for a config, returning list of record lists."""
    fmt = RECALL_FMT.get(source, {}).get(recall)
    if fmt is None:
        return []
    runs = []
    for seed in SEEDS:
        d = base_dir / f"{env_prefix}_{config_key}_{fmt}_s{seed}"
        data = load_run(d)
        if data:
            runs.append(data)
    return runs


def plot_one_grid(frac, recall, seeds_only=False, log_x=True):
    """Plot one grid for a given (frac, recall) combination."""
    configs = FRAC_CONFIGS.get(frac, [])
    n_envs = len(ENVS)
    fig, axes = plt.subplots(n_envs, 3, figsize=(18, 5 * n_envs))
    if n_envs == 1:
        axes = axes[np.newaxis, :]
    scale_label = "log" if log_x else "linear"
    fig.suptitle(f"P3 Comparison — frac={frac}, recall={recall} ({scale_label} x)",
                 fontsize=18, fontweight="bold", y=0.995)
    labels_assigned = set()
    n_found = 0
    n_missing = 0

    for row, (env_name, env_prefix) in enumerate(ENVS):
        metric_keys = None
        all_run_groups = {}

        # 1. Load baselines (from retain_pass_v2, no recall dimension)
        for btype in ["baseline", "filter", "reward_penalty"]:
            prefix = f"{btype}_"
            bl_runs = []
            for seed in SEEDS:
                d = V2_DIR / f"{prefix}{env_prefix}_s{seed}"
                data = load_run(d)
                if data:
                    bl_runs.append(data)
                    if metric_keys is None:
                        metric_keys = discover_metric_keys(data)
                    n_found += 1
                else:
                    n_missing += 1
            if bl_runs:
                all_run_groups[f"bl_{btype}"] = (
                    bl_runs, BASELINE_COLORS[btype], BASELINE_LABELS[btype], "both", "-"
                )

        # 2. Load no_p3 (from retain_pass_v2)
        no_p3_runs = _load_seeds(V2_DIR, env_prefix, "no_p3", recall, "v2")
        if no_p3_runs:
            all_run_groups["no_p3"] = (no_p3_runs, NO_P3_COLOR, NO_P3_LABEL, "retain_only", "-")
            all_run_groups["no_p3_both"] = (no_p3_runs, NO_P3_COLOR, NO_P3_LABEL + " (both)", "both", ":")
            if metric_keys is None and no_p3_runs:
                metric_keys = discover_metric_keys(no_p3_runs[0])
            n_found += len(no_p3_runs)

        # 3. Load P3 selector configs for this frac
        for config_key, base_dir, label, color in configs:
            source = "v2" if base_dir == V2_DIR else "p3"
            runs = _load_seeds(base_dir, env_prefix, config_key, recall, source)
            if runs:
                all_run_groups[config_key] = (runs, color, label, "retain_only", "-")
                all_run_groups[config_key + "_both"] = (runs, color, label + " (both)", "both", ":")
                if metric_keys is None:
                    metric_keys = discover_metric_keys(runs[0])
                n_found += len(runs)
            else:
                n_missing += len(SEEDS)

        if metric_keys is None:
            for ax in axes[row]:
                ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")
            continue

        metric_configs = [
            ("Retain Reward", "retain", (0, 1)),
            ("Hack Frequency", "hack_freq", (0, 1)),
            ("Retain - Hack", "retain_minus_hack", (-1, 1)),
        ]

        for col, (title, metric_id, ylim) in enumerate(metric_configs):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylim(ylim)
            if log_x:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(env_name.replace("_", " ").title(), fontsize=12,
                              fontweight="bold")

            for gkey, (runs, color, label, mode, ls) in all_run_groups.items():
                if mode not in metric_keys:
                    continue
                rk, hk = metric_keys[mode]
                if metric_id == "retain":
                    plot_series(ax, runs, rk, color, label, labels_assigned,
                                linestyle=ls, seeds_only=seeds_only)
                elif metric_id == "hack_freq":
                    plot_series(ax, runs, hk, color, label, labels_assigned,
                                linestyle=ls, seeds_only=seeds_only)
                else:
                    plot_derived_series(ax, runs, rk, hk, color, label,
                                        labels_assigned, linestyle=ls,
                                        seeds_only=seeds_only)

    # Legend
    seen = {}
    for row_axes in axes:
        for ax in row_axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen[l] = h
    # Order: baselines, no_p3, then P3 configs
    ordered = []
    for label in list(BASELINE_LABELS.values()) + [NO_P3_LABEL]:
        if label in seen:
            ordered.append(label)
    for _, _, label, _ in configs:
        if label in seen:
            ordered.append(label)
    ncol = min(len(ordered), 6)
    if ordered:
        fig.legend([seen[l] for l in ordered], ordered, loc="lower center",
                   ncol=ncol, fontsize=11, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig, n_found, n_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_only", action="store_true",
                        help="Show individual seed traces without EMA mean")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_seeds" if args.seeds_only else ""

    # p3_penalty only has recall=0.5; retain_pass_v2 has 0.5, 1.0, 0.1
    # For frac=0.125 we can show all recalls (data from v2 + p3)
    # For frac=0.5, 1.0 we only have recall=0.5
    for log_x, scale_tag in [(True, ""), (False, "_linear")]:
        for frac in [0.125, 0.5, 1.0]:
            recalls = [0.5]
            if frac == 0.125:
                recalls = [0.5, 1.0, 0.1]
            for recall in recalls:
                fig, n_found, n_missing = plot_one_grid(
                    frac, recall, seeds_only=args.seeds_only, log_x=log_x)
                tag = f"frac{str(frac).replace('.', '')}_rcl{str(recall).replace('.', '')}"
                out_path = out_dir / f"p3_comparison_{tag}{scale_tag}{suffix}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[{'log' if log_x else 'linear'}, frac={frac}, recall={recall}] "
                      f"{n_found} found, {n_missing} missing -> {out_path}")


if __name__ == "__main__":
    main()
