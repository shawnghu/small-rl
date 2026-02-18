#!/usr/bin/env python3
"""
plot_routing_comparison.py - Grouped bar charts comparing gradient routing conditions.

For each experiment, shows Proxy Reward, Task Reward, and Happy Frequency across:
  Blue   = No Routing (DualLoRA without gradient routing)
  Orange = Both adapters active
  Red    = Forget adapter only
  Green  = Retain adapter only

Left y-axis: Reward (0-1). Right y-axis: Happy Frequency (normalized, 0-1).

Usage:
    python plot_routing_comparison.py output/my_run --task_key sentence_length_10_smooth \
        --combined_key sentence_length_10_smooth_with_happy --step 1000

Primarily used as a library by sweep_plots.py:
    from plot_routing_comparison import parse_routing_evals, extract_routing_metrics, \
        aggregate_seeds, plot_routing_chart, CONDITION_COLORS, CONDITION_LABELS
"""

import argparse
import os
import re
import statistics
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Data extraction from train logs
# ============================================================

def parse_routing_evals(log_path):
    """Parse routing eval blocks from train.log.

    Returns: {step: {mode: {metric_name: float}}}
    """
    evals = {}
    current_step = None
    if not os.path.exists(log_path):
        return evals
    with open(log_path) as f:
        for line in f:
            m = re.match(r"\[Routing Eval @ step (\d+)\]", line.strip())
            if m:
                current_step = int(m.group(1))
                evals[current_step] = {}
                continue
            if current_step is not None:
                m = re.match(r"\s+(both|retain_only|forget_only)\s+(.*)", line)
                if m:
                    mode = m.group(1)
                    metrics = {}
                    for k, v in re.findall(r"([\w]+)=([\d.]+)", m.group(2)):
                        metrics[k] = float(v)
                    evals[current_step][mode] = metrics
    return evals


def extract_routing_metrics(run_dir, step, task_key, combined_key):
    """Extract routing eval metrics from a run at a given step.

    Returns: {mode: {'combined': float, 'task': float, 'hack_freq': float}}
    or None if no routing eval data. Asserts that combined_key, task_key, and
    hack_freq are all present in the eval log.
    """
    log_path = os.path.join(run_dir, "train.log")
    evals = parse_routing_evals(log_path)
    if not evals:
        return None

    available = sorted(evals.keys())
    target = step if step in evals else min(available, key=lambda s: abs(s - step))

    result = {"_step": target}
    for mode, metrics in evals[target].items():
        assert "hack_freq" in metrics, (
            f"Metric 'hack_freq' missing from eval log at step {target}, mode '{mode}'. "
            f"Available metrics: {list(metrics.keys())}"
        )
        assert combined_key in metrics, (
            f"Metric '{combined_key}' missing from eval log at step {target}, mode '{mode}'. "
            f"Available metrics: {list(metrics.keys())}"
        )
        assert task_key in metrics, (
            f"Metric '{task_key}' missing from eval log at step {target}, mode '{mode}'. "
            f"Available metrics: {list(metrics.keys())}"
        )
        result[mode] = {
            "combined": metrics[combined_key],
            "task": metrics[task_key],
            "hack_freq": metrics["hack_freq"],
        }
    return result


# ============================================================
# Aggregation
# ============================================================

def aggregate_seeds(seed_results):
    """Average metrics across seeds.

    Args: list of {mode: {metric: value}} dicts
    Returns: {mode: {metric: (mean, std)}}
    """
    all_modes = set()
    for r in seed_results:
        all_modes.update(k for k in r if not k.startswith("_"))

    agg = {}
    for mode in all_modes:
        agg[mode] = {}
        for metric in ["combined", "task", "hack_freq"]:
            vals = [r[mode][metric] for r in seed_results if mode in r and metric in r[mode]]
            assert vals, (
                f"Metric '{metric}' missing from all seeds for mode '{mode}'. "
                f"Seed result keys: {[list(r.get(mode, {}).keys()) for r in seed_results if mode in r]}"
            )
            agg[mode][metric] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0)
    return agg


# ============================================================
# Plotting
# ============================================================

CONDITION_COLORS = {
    "baseline":    "#4878CF",  # blue
    "both":        "#E8853B",  # orange
    "forget_only": "#D65F5F",  # red
    "retain_only": "#59A14F",  # green
}

CONDITION_LABELS = {
    "baseline":    "No Routing",
    "both":        "Both Adapters",
    "forget_only": "Forget Only",
    "retain_only": "Retain Only",
}


def plot_routing_chart(
    title,
    data,
    output_path,
    figsize=(9, 5.5),
    step_info="",
    n_seeds=None,
):
    """Generate a grouped bar chart comparing adapter modes.

    Args:
        title: Chart title
        data: {mode: {metric: (mean, std)}}
              mode in ['baseline', 'both', 'forget_only', 'retain_only']
              metric in ['combined', 'task', 'hack_freq']
        output_path: Where to save the PNG
        step_info: Optional annotation (e.g. "step 100, 6 seeds")
        n_seeds: Number of seeds (for annotation)
    """
    metric_labels = ["Proxy Reward", "Task Reward", "Hack Frequency"]
    metric_keys = ["combined", "task", "hack_freq"]

    conditions = [m for m in ["baseline", "both", "forget_only", "retain_only"] if m in data]
    n_cond = len(conditions)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    bar_width = 0.7 / max(n_cond, 1)
    x = np.arange(len(metric_labels))

    for i, cond in enumerate(conditions):
        offset = (i - n_cond / 2 + 0.5) * bar_width
        means, stds = [], []

        for key in metric_keys:
            m, s = data[cond].get(key, (0, 0))
            means.append(m)
            stds.append(s)

        bars = ax1.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=3,
            color=CONDITION_COLORS[cond],
            label=CONDITION_LABELS[cond],
            edgecolor="white", linewidth=0.8, alpha=0.9,
            error_kw={"linewidth": 1.2},
        )

        # Hatch happy frequency bar to visually distinguish
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor("gray")

        # Value labels
        for bi, (bar, mean_val) in enumerate(zip(bars, means)):
            if mean_val > 0.03:
                y_pos = bar.get_height() + stds[bi] + 0.02
                if y_pos > 1.05:
                    y_pos = bar.get_height() - 0.05
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos, f"{mean_val:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                )

    ax1.set_ylabel("Reward", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Hack Frequency (fraction of samples)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.18)
    ax2.set_ylim(0, 1.18)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11)

    subtitle = step_info
    if n_seeds:
        subtitle = f"{n_seeds} seeds, {subtitle}" if subtitle else f"{n_seeds} seeds"
    full_title = title
    if subtitle:
        full_title += f"\n({subtitle})"
    ax1.set_title(full_title, fontsize=13, fontweight="bold")

    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot routing comparison chart for a single run")
    parser.add_argument("run_dir", help="Path to a run directory containing train.log")
    parser.add_argument("--task_key", required=True, help="Metric key for task reward (e.g. sentence_length_10_smooth)")
    parser.add_argument("--combined_key", required=True, help="Metric key for combined reward (e.g. sentence_length_10_smooth_with_happy)")
    parser.add_argument("--step", type=int, default=None, help="Eval step to plot (default: latest)")
    parser.add_argument("--output", default="routing_chart.png", help="Output PNG path")
    args = parser.parse_args()

    step = args.step or 10**9
    data = extract_routing_metrics(args.run_dir, step, args.task_key, args.combined_key)
    if not data:
        print(f"No routing eval data found in {args.run_dir}")
        sys.exit(1)

    agg = aggregate_seeds([data])
    actual_step = data.get("_step", step)
    plot_routing_chart(
        os.path.basename(args.run_dir.rstrip("/\\")) ,
        agg, args.output,
        step_info=f"step {actual_step}",
    )


if __name__ == "__main__":
    main()
