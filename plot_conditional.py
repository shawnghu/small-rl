"""Plot routing eval metrics for the conditional detector sweep.

Generates a 5×3 grid (5 envs × 3 metrics) for each retain mode,
comparing routing adapter modes against baselines (do-nothing, filter,
reward penalty). Baseline type is identified from run_config.yaml since
all three share the same directory name pattern (dir collision).

Usage:
    uv run python plot_conditional.py                    # both retain modes
    uv run python plot_conditional.py --retain_mode renormalize
    uv run python plot_conditional.py --retain_mode penalty
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

BASE_DIR = Path("output/routing_conditional_detectors")

ENVS = [
    ("addition_v2", "addition_v2_sycophancy_conditional_ms10000"),
    ("cities_qa", "cities_qa_sycophancy_conditional_ms5000"),
    ("persona_qa", "persona_qa_flattery_conditional_ms5000"),
    ("repeat", "repeat_extra_conditional_ms1000"),
    ("topic", "topic_contains_conditional_ms1000"),
]

RETAIN_MODES = {
    "renormalize": {"suffix": "retmrenormalize_retpmissing_rmexclusive"},
    "penalty": {"suffix": "retmpenalty_retp2.0_rmexclusive"},
}

BASELINE_SUFFIX = "retmmissing_retpmissing_rmnone"

SEEDS = [1, 2, 3]

COLORS = {
    "retain_only": "#59A14F",
    "forget_only": "#D65F5F",
    "both": "#E8853B",
    "baseline": "#4878CF",
    "filter": "#9B59B6",
    "reward_penalty": "#8B4513",
}

LABELS = {
    "retain_only": "Retain Only",
    "forget_only": "Forget Only",
    "both": "Both Adapters",
    "baseline": "Do-Nothing Baseline",
    "filter": "Filter (drop RH)",
    "reward_penalty": "Reward Penalty",
}


def classify_baseline(run_dir):
    """Read run_config.yaml to determine baseline type: 'baseline', 'filter', or 'reward_penalty'.

    Keys are nested under 'training:' in run_config.yaml.
    """
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        return "baseline"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    training = cfg.get("training", {})
    if training.get("filter_baseline"):
        return "filter"
    if training.get("reward_penalty_baseline"):
        return "reward_penalty"
    return "baseline"


def load_jsonl(path):
    by_step = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            by_step[record["step"]] = record
    return [by_step[s] for s in sorted(by_step)]


def ema(values, alpha=0.15):
    result = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        result.append(s)
    return result


def find_metric_key(keys, mode, metric_type):
    prefix = f"{mode}/{metric_type}/"
    matches = [k for k in keys if k.startswith(prefix)]
    assert len(matches) == 1, f"Expected 1 match for {prefix}, got {matches}"
    return matches[0]


def load_run(run_dir):
    path = run_dir / "routing_eval.jsonl"
    assert path.exists(), f"Missing {path}"
    return load_jsonl(path)


def extract_series(records, key):
    steps, values = [], []
    for r in records:
        if r["step"] == 0:
            continue
        if key in r:
            steps.append(r["step"])
            values.append(r[key])
    return np.array(steps), np.array(values)


def plot_grid(retain_mode, out_dir):
    """Generate a 5×3 grid for one retain mode."""
    rm_info = RETAIN_MODES[retain_mode]
    fig, axes = plt.subplots(5, 3, figsize=(18, 25))
    fig.suptitle(f"Conditional Detectors — retain_mode={retain_mode}",
                 fontsize=18, fontweight="bold", y=0.995)
    # Track which labels have been assigned (for legend dedup)
    labels_assigned = set()

    for row, (env_name, env_prefix) in enumerate(ENVS):
        # Load routing runs (3 seeds)
        routing_runs = []
        for seed in SEEDS:
            d = BASE_DIR / f"{env_prefix}_{rm_info['suffix']}_s{seed}"
            if not (d / "routing_eval.jsonl").exists():
                print(f"  SKIP (missing): {d.name}")
                continue
            routing_runs.append(load_run(d))

        if not routing_runs:
            print(f"  No routing runs for {env_name} / {retain_mode}")
            for ax in axes[row]:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")
            continue

        # Load baseline runs (3 seeds), classify by type from run_config.yaml
        baseline_by_type = {"baseline": [], "filter": [], "reward_penalty": []}
        for seed in SEEDS:
            d = BASE_DIR / f"{env_prefix}_{BASELINE_SUFFIX}_s{seed}"
            if not (d / "routing_eval.jsonl").exists():
                print(f"  SKIP baseline (missing): {d.name}")
                continue
            btype = classify_baseline(d)
            baseline_by_type[btype].append(load_run(d))

        # Discover metric keys
        sample_keys = [k for k in routing_runs[0][0].keys() if k != "step"]
        mode_keys = {}
        for mode in ["both", "retain_only", "forget_only"]:
            mode_keys[mode] = (
                find_metric_key(sample_keys, mode, "retain"),
                find_metric_key(sample_keys, mode, "hack_freq"),
            )

        retain_key_both = mode_keys["both"][0]
        hack_key_both = mode_keys["both"][1]

        metric_configs = [
            ("Retain Reward", "retain", (0, 1)),
            ("Hack Frequency", "hack_freq", (0, 1)),
            ("Retain − Hack", "retain_minus_hack", (-1, 1)),
        ]

        for col, (title, metric_id, ylim) in enumerate(metric_configs):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylim(ylim)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(env_name.replace("_", " ").title(), fontsize=12,
                              fontweight="bold")

            # Plot routing conditions
            for mode in ["retain_only", "forget_only", "both"]:
                rk, hk = mode_keys[mode]
                all_steps, all_values = [], []

                for run_data in routing_runs:
                    if metric_id == "retain":
                        steps, vals = extract_series(run_data, rk)
                    elif metric_id == "hack_freq":
                        steps, vals = extract_series(run_data, hk)
                    else:
                        steps, r_vals = extract_series(run_data, rk)
                        _, h_vals = extract_series(run_data, hk)
                        vals = r_vals - h_vals

                    ax.plot(steps, vals, color=COLORS[mode], alpha=0.12, linewidth=0.6)
                    all_steps.append(steps)
                    all_values.append(vals)

                # EMA of mean across seeds
                min_len = min(len(v) for v in all_values)
                if min_len == 0:
                    continue
                ref_steps = all_steps[0][:min_len]
                truncated = [v[:min_len] for v in all_values]
                mean_vals = np.mean(truncated, axis=0)
                ema_vals = ema(mean_vals.tolist())
                label = LABELS[mode] if mode not in labels_assigned else None
                if label:
                    labels_assigned.add(mode)
                ax.plot(ref_steps, ema_vals, color=COLORS[mode], linewidth=2.5,
                        label=label)

            # Plot baselines by type (do-nothing, filter, reward_penalty)
            for btype in ["baseline", "filter", "reward_penalty"]:
                bl_runs = baseline_by_type[btype]
                if not bl_runs:
                    continue
                bl_all_steps, bl_all_values = [], []
                for run_data in bl_runs:
                    if metric_id == "retain":
                        steps, vals = extract_series(run_data, retain_key_both)
                    elif metric_id == "hack_freq":
                        steps, vals = extract_series(run_data, hack_key_both)
                    else:
                        steps, r_vals = extract_series(run_data, retain_key_both)
                        _, h_vals = extract_series(run_data, hack_key_both)
                        vals = r_vals - h_vals

                    ax.plot(steps, vals, color=COLORS[btype], alpha=0.08,
                            linewidth=0.5)
                    bl_all_steps.append(steps)
                    bl_all_values.append(vals)

                min_len = min(len(v) for v in bl_all_values)
                if min_len > 0:
                    ref_steps = bl_all_steps[0][:min_len]
                    truncated = [v[:min_len] for v in bl_all_values]
                    mean_vals = np.mean(truncated, axis=0)
                    ema_vals = ema(mean_vals.tolist())
                    label = LABELS[btype] if btype not in labels_assigned else None
                    if label:
                        labels_assigned.add(btype)
                    ax.plot(ref_steps, ema_vals, color=COLORS[btype],
                            linewidth=2.5, label=label)

    # Collect legend handles from all axes (dedup by label)
    seen = {}
    for row_axes in axes:
        for ax in row_axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen[l] = h
    # Stable order matching LABELS
    ordered_labels = [l for l in LABELS.values() if l in seen]
    ordered_handles = [seen[l] for l in ordered_labels]
    ncol = min(len(ordered_handles), 6)
    fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=ncol,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path = out_dir / f"conditional_{retain_mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retain_mode", choices=["renormalize", "penalty"],
                        default=None, help="Plot one retain mode (default: both)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: {BASE_DIR}/slides)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "slides"
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [args.retain_mode] if args.retain_mode else list(RETAIN_MODES.keys())
    for rm in modes:
        print(f"\n=== Plotting retain_mode={rm} ===")
        plot_grid(rm, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
