"""All routing configs on one plot: no_p3, nondetected, and penalized across frac levels.

All lines use retain_only adapter mode. 10 lines total per env×metric:
  - no_p3 (orange)
  - nondet f=0.125 (light blue), f=0.5 (medium blue), f=1.0 (dark blue)
  - pen f=0.125 (light green), f=0.5 (medium green), f=1.0 (dark green)
  - pen f=1.0 rcl=0.1 (green), pen f=2.0 rcl=0.5 (deep purple), pen f=2.0 rcl=0.1 (medium purple)

Usage:
    uv run python plot_p3_all_routing.py
    uv run python plot_p3_all_routing.py --seeds_only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_retain_pass import (
    SEEDS,
    discover_metric_keys, load_run,
    plot_series, plot_derived_series,
)

V2_DIR = Path("output/retain_pass_v2")
P3_DIR = Path("output/p3_penalty")
P3_HF_DIR = Path("output/p3_high_frac")
OUT_DIR = Path("output/p3_all_routing")

ENVS = [
    ("addition_v2", "addition_v2_sycophancy_conditional"),
    ("cities_qa", "cities_qa_sycophancy_conditional"),
    ("persona_qa", "persona_qa_flattery_conditional"),
    ("repeat", "repeat_extra_conditional"),
    ("topic", "topic_contains_conditional"),
]

# (config_key, base_dir, recall_fmt, label, color)
# recall_fmt: how recall=0.5 is encoded in directory names
LINE_DEFS = [
    ("no_p3",        V2_DIR, "rcl0.5", "No P3",                "#E8853B"),  # orange
    ("fresh_nondet", V2_DIR, "rcl0.5", "Nondetected (f=0.125)", "#90CAF9"),  # light blue
    ("nondet_f05",   P3_DIR, "rcl05",  "Nondetected (f=0.5)",   "#42A5F5"),  # medium blue
    ("nondet_f10",   P3_DIR, "rcl05",  "Nondetected (f=1.0)",   "#1565C0"),  # dark blue
    ("pen_f0125",    P3_DIR, "rcl05",  "Penalized (f=0.125)",   "#81C784"),  # light green
    ("pen_f05",      P3_DIR, "rcl05",  "Penalized (f=0.5)",     "#388E3C"),  # medium green
    ("pen_f10",      P3_DIR, "rcl05",  "Penalized (f=1.0)",     "#1B5E20"),  # dark green
    ("pen_f10_rcl01", P3_HF_DIR, "rcl01", "Penalized (f=1.0, r=0.1)", "#2E7D32"),  # green
    ("pen_f20_rcl05", P3_HF_DIR, "rcl05", "Penalized (f=2.0, r=0.5)", "#4A148C"),  # deep purple
    ("pen_f20_rcl01", P3_HF_DIR, "rcl01", "Penalized (f=2.0, r=0.1)", "#7B1FA2"),  # medium purple
]


def _load_seeds(base_dir, env_prefix, config_key, recall_fmt):
    """Load seed runs for a config, returning list of record lists."""
    runs = []
    for seed in SEEDS:
        d = base_dir / f"{env_prefix}_{config_key}_{recall_fmt}_s{seed}"
        data = load_run(d)
        if data:
            runs.append(data)
    return runs


def plot_all_routing(seeds_only=False, log_x=True):
    """Plot a single grid: N_envs x 3 metrics with all routing lines."""
    n_envs = len(ENVS)
    fig, axes = plt.subplots(n_envs, 3, figsize=(18, 5 * n_envs))
    if n_envs == 1:
        axes = axes[np.newaxis, :]
    scale_label = "log" if log_x else "linear"
    fig.suptitle(f"All Routing Configs — recall=0.5, retain_only ({scale_label} x)",
                 fontsize=18, fontweight="bold", y=0.995)
    labels_assigned = set()
    n_found = 0
    n_missing = 0

    for row, (env_name, env_prefix) in enumerate(ENVS):
        metric_keys = None
        run_groups = {}  # key -> (runs, color, label)

        for config_key, base_dir, recall_fmt, label, color in LINE_DEFS:
            runs = _load_seeds(base_dir, env_prefix, config_key, recall_fmt)
            if runs:
                run_groups[config_key] = (runs, color, label)
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
            ("Retain \u2212 Hack", "retain_minus_hack", (-1, 1)),
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

            mode = "retain_only"
            if mode not in metric_keys:
                continue
            rk, hk = metric_keys[mode]

            for config_key, base_dir, recall_fmt, label, color in LINE_DEFS:
                if config_key not in run_groups:
                    continue
                runs, color, label = run_groups[config_key]

                if metric_id == "retain":
                    plot_series(ax, runs, rk, color, label, labels_assigned,
                                seeds_only=seeds_only)
                elif metric_id == "hack_freq":
                    plot_series(ax, runs, hk, color, label, labels_assigned,
                                seeds_only=seeds_only)
                else:
                    plot_derived_series(ax, runs, rk, hk, color, label,
                                        labels_assigned, seeds_only=seeds_only)

    # Legend — preserve LINE_DEFS order
    seen = {}
    for row_axes in axes:
        for ax in row_axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen[l] = h
    ordered = [label for _, _, _, label, _ in LINE_DEFS if label in seen]
    ncol = min(len(ordered), 5)
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

    for log_x, tag in [(True, ""), (False, "_linear")]:
        fig, n_found, n_missing = plot_all_routing(
            seeds_only=args.seeds_only, log_x=log_x)
        out_path = out_dir / f"p3_all_routing{tag}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{'log' if log_x else 'linear'}] Found {n_found} runs, "
              f"{n_missing} missing -> {out_path}")


if __name__ == "__main__":
    main()
