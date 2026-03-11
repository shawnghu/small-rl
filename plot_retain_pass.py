"""Plot routing eval metrics for retain_pass sweeps.

Supports both v1 (retain_pass_conditional, no recall) and v2 (retain_pass_v2, recall sweep).

For v1 (no recall): generates a N_envs×3 grid (envs × metrics).
For v2 (with recall): generates a N_envs×3 grid per recall value, or a faceted grid.

Usage:
    uv run python plot_retain_pass.py --base_dir output/retain_pass_conditional
    uv run python plot_retain_pass.py --base_dir output/retain_pass_v2
    uv run python plot_retain_pass.py --base_dir output/retain_pass_v2 --recall 1.0 0.5 0.1
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

BASE_DIR = Path("output/retain_pass_v2")

ENVS = [
    ("addition_v2", "addition_v2_sycophancy_conditional"),
    ("cities_qa", "cities_qa_sycophancy_conditional"),
    ("persona_qa", "persona_qa_flattery_conditional"),
    ("repeat", "repeat_extra_conditional"),
    ("topic", "topic_contains_conditional"),
    ("sorting", "sorting_sycophancy_persona_conditional"),
]

ROUTING_CONFIGS = ["no_p3", "fresh_random", "fresh_nondet"]

SEEDS = [1, 2, 3]

# Colors for routing configs (retain_only mode)
ROUTING_COLORS = {
    "no_p3": "#E8853B",         # orange
    "fresh_random": "#59A14F",  # green
    "fresh_nondet": "#6B5B95",  # purple
}

ROUTING_LABELS = {
    "no_p3": "Routing (no Phase 3)",
    "fresh_random": "Routing + Fresh Random",
    "fresh_nondet": "Routing + Fresh Nondetected",
}

# Colors for baselines
BASELINE_COLORS = {
    "baseline": "#4878CF",    # blue
    "filter": "#9B59B6",      # purple
    "reward_penalty": "#8B4513",  # brown
}

BASELINE_LABELS = {
    "baseline": "Do-Nothing Baseline",
    "filter": "Filter (drop RH)",
    "reward_penalty": "Reward Penalty",
}

# Extra adapter mode colors (for --show_all_modes)
MODE_SUFFIX_COLORS = {
    "both": {"alpha_mult": 0.5, "linestyle": "--"},
    "forget_only": {"alpha_mult": 0.3, "linestyle": ":"},
}


def load_jsonl(path):
    by_step = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            by_step[record["step"]] = record
    return [by_step[s] for s in sorted(by_step)]


def ema(values, alpha=0.5):
    if not values:
        return []
    result = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        result.append(s)
    return result


def find_metric_key(keys, mode, metric_type):
    prefix = f"{mode}/{metric_type}/"
    matches = [k for k in keys if k.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    # Fallback: exact match (e.g., "retain_only/hack_freq")
    exact = f"{mode}/{metric_type}"
    if exact in keys:
        return exact
    return None


def load_run(run_dir):
    path = run_dir / "routing_eval.jsonl"
    if not path.exists():
        return None
    records = load_jsonl(path)
    return records if records else None


def extract_series(records, key):
    steps, values = [], []
    for r in records:
        if r["step"] == 0:
            continue
        if key in r:
            steps.append(r["step"])
            values.append(r[key])
    return np.array(steps), np.array(values)


def classify_baseline(run_dir):
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


def discover_metric_keys(records):
    """Discover retain and hack_freq keys from first record."""
    sample_keys = [k for k in records[0].keys() if k != "step"]
    keys = {}
    for mode in ["both", "retain_only", "forget_only"]:
        rk = find_metric_key(sample_keys, mode, "retain")
        hk = find_metric_key(sample_keys, mode, "hack_freq")
        if rk and hk:
            keys[mode] = (rk, hk)
    return keys


def plot_series(ax, runs_data, key, color, label, labels_assigned, linestyle="-"):
    """Plot individual seed traces + EMA mean for a set of seed runs."""
    all_steps, all_values = [], []
    for run_data in runs_data:
        steps, vals = extract_series(run_data, key)
        if len(steps) == 0:
            continue
        ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.6,
                linestyle=linestyle)
        all_steps.append(steps)
        all_values.append(vals)

    if not all_values:
        return

    min_len = min(len(v) for v in all_values)
    if min_len == 0:
        return
    ref_steps = all_steps[0][:min_len]
    truncated = [v[:min_len] for v in all_values]
    mean_vals = np.mean(truncated, axis=0)
    ema_vals = ema(mean_vals.tolist())
    lbl = label if label not in labels_assigned else None
    if lbl:
        labels_assigned.add(label)
    ax.plot(ref_steps, ema_vals, color=color, linewidth=2.5, label=lbl,
            linestyle=linestyle)


def plot_derived_series(ax, runs_data, rkey, hkey, color, label, labels_assigned,
                        linestyle="-"):
    """Plot retain - hack for a set of seed runs."""
    all_steps, all_values = [], []
    for run_data in runs_data:
        steps, r_vals = extract_series(run_data, rkey)
        _, h_vals = extract_series(run_data, hkey)
        min_l = min(len(r_vals), len(h_vals))
        if min_l == 0:
            continue
        vals = r_vals[:min_l] - h_vals[:min_l]
        ax.plot(steps[:min_l], vals, color=color, alpha=0.12, linewidth=0.6,
                linestyle=linestyle)
        all_steps.append(steps[:min_l])
        all_values.append(vals)

    if not all_values:
        return

    min_len = min(len(v) for v in all_values)
    if min_len == 0:
        return
    ref_steps = all_steps[0][:min_len]
    truncated = [v[:min_len] for v in all_values]
    mean_vals = np.mean(truncated, axis=0)
    ema_vals = ema(mean_vals.tolist())
    lbl = label if label not in labels_assigned else None
    if lbl:
        labels_assigned.add(label)
    ax.plot(ref_steps, ema_vals, color=color, linewidth=2.5, label=lbl,
            linestyle=linestyle)


def _plot_one_grid(base_dir, envs, recall=None, show_all_modes=False):
    """Plot a single grid for a given recall value (or None for v1 format)."""
    n_envs = len(envs)
    fig, axes = plt.subplots(n_envs, 3, figsize=(18, 5 * n_envs))
    if n_envs == 1:
        axes = axes[np.newaxis, :]
    recall_str = f" (recall={recall})" if recall is not None else ""
    fig.suptitle(f"Retain Pass Sweep — retain_only vs Baselines{recall_str}",
                 fontsize=18, fontweight="bold", y=0.995)
    labels_assigned = set()

    n_found = 0
    n_missing = 0

    for row, (env_name, env_prefix) in enumerate(envs):
        # Load routing runs by config
        routing_by_config = {}
        metric_keys = None
        for rc in ROUTING_CONFIGS:
            runs = []
            for seed in SEEDS:
                if recall is not None:
                    d = base_dir / f"{env_prefix}_{rc}_rcl{recall}_s{seed}"
                else:
                    d = base_dir / f"{env_prefix}_{rc}_s{seed}"
                data = load_run(d)
                if data:
                    runs.append(data)
                    if metric_keys is None:
                        metric_keys = discover_metric_keys(data)
                    n_found += 1
                else:
                    n_missing += 1
            if runs:
                routing_by_config[rc] = runs

        # Load baselines (no recall dimension)
        baseline_by_type = {"baseline": [], "filter": [], "reward_penalty": []}
        for seed in SEEDS:
            for prefix, btype in [("baseline_", "baseline"), ("filter_", "filter"),
                                   ("reward_penalty_", "reward_penalty")]:
                d = base_dir / f"{prefix}{env_prefix}_s{seed}"
                data = load_run(d)
                if data:
                    baseline_by_type[btype].append(data)
                    if metric_keys is None:
                        metric_keys = discover_metric_keys(data)
                    n_found += 1
                else:
                    n_missing += 1

        if metric_keys is None:
            for ax in axes[row]:
                ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")
            continue

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

            # Plot each routing config in retain_only (solid) and both (dotted)
            for rc, runs in routing_by_config.items():
                color = ROUTING_COLORS[rc]
                for mode, linestyle in [("retain_only", "-"), ("both", ":")]:
                    if mode not in metric_keys:
                        continue
                    rk, hk = metric_keys[mode]
                    suffix = "" if mode == "retain_only" else " (both)"
                    label = ROUTING_LABELS[rc] + suffix

                    if metric_id == "retain":
                        plot_series(ax, runs, rk, color, label,
                                    labels_assigned, linestyle=linestyle)
                    elif metric_id == "hack_freq":
                        plot_series(ax, runs, hk, color, label,
                                    labels_assigned, linestyle=linestyle)
                    else:
                        plot_derived_series(ax, runs, rk, hk, color, label,
                                            labels_assigned, linestyle=linestyle)

            # Plot baselines (both mode since no routing)
            bl_mode = "both"
            if bl_mode in metric_keys:
                bl_rk, bl_hk = metric_keys[bl_mode]
                for btype in ["baseline", "filter", "reward_penalty"]:
                    bl_runs = baseline_by_type[btype]
                    if not bl_runs:
                        continue
                    color = BASELINE_COLORS[btype]
                    label = BASELINE_LABELS[btype]

                    if metric_id == "retain":
                        plot_series(ax, bl_runs, bl_rk, color, label,
                                    labels_assigned)
                    elif metric_id == "hack_freq":
                        plot_series(ax, bl_runs, bl_hk, color, label,
                                    labels_assigned)
                    else:
                        plot_derived_series(ax, bl_runs, bl_rk, bl_hk, color,
                                            label, labels_assigned)

    # Legend
    seen = {}
    for row_axes in axes:
        for ax in row_axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen[l] = h
    all_labels = list(ROUTING_LABELS.values()) + list(BASELINE_LABELS.values())
    ordered_labels = [l for l in all_labels if l in seen]
    ordered_handles = [seen[l] for l in ordered_labels]
    ncol = min(len(ordered_handles), 6)
    if ordered_handles:
        fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=ncol,
                   fontsize=11, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig, n_found, n_missing


def plot_grid(base_dir, recall_values=None, show_all_modes=False):
    """Plot grids — one per recall value if specified, else a single grid."""
    out_dir = base_dir / "slides"
    out_dir.mkdir(parents=True, exist_ok=True)

    if recall_values:
        for recall in recall_values:
            fig, n_found, n_missing = _plot_one_grid(
                base_dir, ENVS, recall=recall, show_all_modes=show_all_modes)
            recall_tag = str(recall).replace(".", "")
            out_path = out_dir / f"retain_pass_comparison_rcl{recall_tag}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[recall={recall}] Found {n_found} runs, {n_missing} missing. Saved {out_path}")
    else:
        fig, n_found, n_missing = _plot_one_grid(
            base_dir, ENVS, recall=None, show_all_modes=show_all_modes)
        out_path = out_dir / "retain_pass_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Found {n_found} runs with data, {n_missing} missing/no eval data")
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=None,
                        help="Sweep output directory")
    parser.add_argument("--recall", nargs="*", type=float, default=None,
                        help="Recall values to plot (generates one grid per value). "
                             "Omit for v1-style single grid (no recall in run names).")
    parser.add_argument("--show_all_modes", action="store_true",
                        help="Also show both/forget_only adapter modes")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    plot_grid(base_dir, recall_values=args.recall, show_all_modes=args.show_all_modes)


if __name__ == "__main__":
    main()
