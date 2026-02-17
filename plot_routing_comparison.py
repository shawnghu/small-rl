#!/usr/bin/env python3
"""
plot_routing_comparison.py - Grouped bar charts comparing gradient routing conditions.

For each experiment, shows Total Reward, Task Reward, and Happy Frequency across:
  Blue   = Baseline (non-routed LoRA)
  Orange = Both adapters active
  Red    = Forget adapter only
  Green  = Retain adapter only

Left y-axis: Reward (0-1). Right y-axis: Happy Frequency (normalized, 0-1).

Usage:
    python plot_routing_comparison.py                     # Pre-configured experiments
    python plot_routing_comparison.py --output_dir plots  # Custom output dir
    python plot_routing_comparison.py --run_eval           # Also run eval on baselines

Can be imported for use in train.py eval pipeline:
    from plot_routing_comparison import plot_routing_chart
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

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
    or None if no routing eval data.

    Uses 'hack_freq' if present. Falls back to normalized happy_count
    (min(count/5, 1)) for older runs that predate the hack_freq metric.
    """
    log_path = os.path.join(run_dir, "train.log")
    evals = parse_routing_evals(log_path)
    if not evals:
        return None

    available = sorted(evals.keys())
    target = step if step in evals else min(available, key=lambda s: abs(s - step))

    result = {"_step": target}
    for mode, metrics in evals[target].items():
        if "hack_freq" in metrics:
            hf = metrics["hack_freq"]
        else:
            # Legacy fallback for old train logs
            hf = min(metrics.get("happy_count", 0) / 5.0, 1.0)
        result[mode] = {
            "combined": metrics.get(combined_key, 0),
            "task": metrics.get(task_key, 0),
            "hack_freq": hf,
        }
    return result


def eval_checkpoint(run_dir, combined_key, task_key, n_samples=20):
    """Run eval on a checkpoint to get decomposed metrics.

    Uses eval_gradient_routing which handles both PEFT and DualLoRA models.
    Results cached in .eval_cache.json.

    Returns: {mode: {'combined': float, 'task': float, 'hack_freq': float}}
             For non-DualLoRA models, mode='baseline'. For DualLoRA, all 3 modes.
    """
    cache_path = os.path.join(run_dir, ".eval_cache.json")
    cache_id = f"{combined_key},{task_key},{n_samples}"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get("_id") == cache_id:
            return cache["data"]

    checkpoints = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")))
    if not checkpoints:
        return None
    checkpoint = checkpoints[-1]

    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from eval_run import load_gradient_routing_model, eval_gradient_routing
    from rewards import REWARD_REGISTRY
    from transformers import AutoTokenizer

    print(f"    Eval {os.path.basename(run_dir)}/{os.path.basename(checkpoint)}...")
    model = load_gradient_routing_model(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")

    reward_fns = {}
    for key in [combined_key, task_key]:
        if key in REWARD_REGISTRY:
            reward_fns[key] = REWARD_REGISTRY[key]
    from rh_detectors import get_rh_detector
    from rewards import make_hack_frequency_fn
    reward_fns["hack_freq"] = make_hack_frequency_fn(get_rh_detector("happy_count", threshold=3))

    results = eval_gradient_routing(model, tokenizer, reward_fns, n_samples=n_samples)

    # Convert to our format
    def _get_mean(metrics, key):
        v = metrics.get(key, {})
        return v.get("mean", 0) if isinstance(v, dict) else 0

    data = {}
    for mode_name, mode_data in results.items():
        metrics = mode_data.get("metrics", {})
        # For non-DualLoRA models, rename "both" -> "baseline"
        out_mode = "baseline" if mode_name == "both" and len(results) == 1 else mode_name
        data[out_mode] = {
            "combined": _get_mean(metrics, combined_key),
            "task": _get_mean(metrics, task_key),
            "hack_freq": _get_mean(metrics, "hack_freq"),
        }

    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(cache_path, "w") as f:
        json.dump({"_id": cache_id, "data": data}, f)
    return data


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
            if vals:
                agg[mode][metric] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0)
            else:
                agg[mode][metric] = (0, 0)
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
    "baseline":    "Baseline (LoRA)",
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
    metric_labels = ["Total Reward", "Task Reward", "Hack Frequency"]
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
# Experiment data collection
# ============================================================

def collect_routing_data(run_pattern, seeds, step, task_key, combined_key):
    """Collect routing eval data across seeds.

    Args:
        run_pattern: glob pattern with {seed} placeholder
        seeds: list of seed values to try
        step: target eval step
        task_key, combined_key: metric keys

    Returns: {mode: {metric: (mean, std)}}, actual_n_seeds
    """
    seed_results = []
    for seed in seeds:
        run_dir = os.path.join(OUTPUT_DIR, run_pattern.format(seed=seed))
        data = extract_routing_metrics(run_dir, step, task_key, combined_key)
        if data:
            seed_results.append(data)

    if not seed_results:
        return {}, 0
    return aggregate_seeds(seed_results), len(seed_results)


def collect_baseline_data(run_pattern, seeds, combined_key, task_key, run_eval=False, n_samples=20):
    """Collect baseline metrics across seeds.

    If run_eval=True, loads checkpoints and computes metrics.
    Otherwise tries to read cached eval data.

    Returns: {mode: {metric: (mean, std)}}, n_seeds
    """
    seed_results = []
    for seed in seeds:
        run_dir = os.path.join(OUTPUT_DIR, run_pattern.format(seed=seed))
        if not os.path.exists(run_dir):
            continue

        # Try cache first
        cache_path = os.path.join(run_dir, ".eval_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
            if "data" in cache and "baseline" in cache["data"]:
                seed_results.append(cache["data"])
                continue

        if run_eval:
            data = eval_checkpoint(run_dir, combined_key, task_key, n_samples)
            if data:
                seed_results.append(data)

    if not seed_results:
        return {}, 0
    return aggregate_seeds(seed_results), len(seed_results)


# ============================================================
# Pre-configured experiments
# ============================================================

SEEDS_6 = [42, 123, 7, 99, 200, 301]
SEEDS_3 = [42, 123, 7]

EXPERIMENTS = [
    {
        "name": "SL5 Gradient Routing (Shared Mode)",
        "filename": "sl5_routing_comparison.png",
        "routing_pattern": "sentence_length_5_with_happy_lcr32_rh0.5_s{seed}",
        "baseline_pattern": None,  # No proper SL5 baseline available
        "seeds": SEEDS_6,
        "step": 600,
        "task_key": "sentence_length_5",
        "combined_key": "sentence_length_5_with_happy",
    },
    {
        "name": "SL10 Gradient Routing (Shared Mode)",
        "filename": "sl10_routing_comparison.png",
        "routing_pattern": "sentence_length_10_smooth_with_happy_lcr32_rh0.5_s{seed}",
        "baseline_pattern": "sentence_length_10_smooth_with_happy_lor64_s{seed}",
        "seeds": SEEDS_6,
        "baseline_seeds": SEEDS_3,
        "step": 100,
        "task_key": "sentence_length_10_smooth",
        "combined_key": "sentence_length_10_smooth_with_happy",
    },
    {
        "name": "SL10 Exclusive Routing (No Ablation)",
        "filename": "sl10_exclusive_comparison.png",
        "routing_pattern": "sentence_length_10_smooth_with_happy_ms1000_rmexclusive_s{seed}",
        "baseline_pattern": "sentence_length_10_smooth_with_happy_lor64_s{seed}",
        "seeds": SEEDS_3,
        "baseline_seeds": SEEDS_3,
        "step": 1000,
        "task_key": "sentence_length_10_smooth",
        "combined_key": "sentence_length_10_smooth_with_happy",
    },
    {
        "name": "SL10 Exclusive Routing + Ablation (Verified-Good Data)",
        "filename": "sl10_exclusive_ablated_comparison.png",
        "routing_pattern": "sentence_length_10_smooth_with_happy_af0.5_ms1000_rmexclusive_s{seed}",
        "baseline_pattern": "sentence_length_10_smooth_with_happy_lor64_s{seed}",
        "seeds": SEEDS_3,
        "baseline_seeds": SEEDS_3,
        "step": 1000,
        "task_key": "sentence_length_10_smooth",
        "combined_key": "sentence_length_10_smooth_with_happy",
    },
]


def run_experiments(output_dir, run_eval=False):
    """Run all pre-configured experiments and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  {exp['name']}")
        print(f"{'='*60}")

        # Collect routing data
        routing_data, n_routing = collect_routing_data(
            exp["routing_pattern"], exp["seeds"],
            exp["step"], exp["task_key"], exp["combined_key"],
        )
        print(f"  Routing: {n_routing} seeds")
        for mode in ["both", "retain_only", "forget_only"]:
            if mode in routing_data:
                m = routing_data[mode]
                print(f"    {mode:15s}  combined={m['combined'][0]:.3f}  task={m['task'][0]:.3f}  hack_freq={m['hack_freq'][0]:.3f}")

        # Collect baseline data
        baseline_data = {}
        n_baseline = 0
        if exp.get("baseline_pattern"):
            baseline_seeds = exp.get("baseline_seeds", exp["seeds"])
            baseline_data, n_baseline = collect_baseline_data(
                exp["baseline_pattern"], baseline_seeds,
                exp["combined_key"], exp["task_key"],
                run_eval=run_eval,
            )
            print(f"  Baseline: {n_baseline} seeds")
            if "baseline" in baseline_data:
                m = baseline_data["baseline"]
                print(f"    {'baseline':15s}  combined={m['combined'][0]:.3f}  task={m['task'][0]:.3f}  hack_freq={m['hack_freq'][0]:.3f}")

        # Merge data
        plot_data = {}
        if "baseline" in baseline_data:
            plot_data["baseline"] = baseline_data["baseline"]
        plot_data.update(routing_data)

        if not plot_data:
            print("  No data available, skipping.")
            continue

        # Determine step info
        step_info = f"step {exp['step']}"
        n_seeds = max(n_routing, n_baseline)

        output_path = os.path.join(output_dir, exp["filename"])
        plot_routing_chart(
            exp["name"], plot_data, output_path,
            step_info=step_info, n_seeds=n_seeds,
        )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot routing comparison charts")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "plots"),
                        help="Directory for output PNGs (default: plots/)")
    parser.add_argument("--run_eval", action="store_true",
                        help="Run eval on baseline checkpoints (slow, ~1 min per model)")
    args = parser.parse_args()
    run_experiments(args.output_dir, run_eval=args.run_eval)


if __name__ == "__main__":
    main()
