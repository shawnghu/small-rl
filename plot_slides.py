"""Generate presentation-quality gradient routing plots."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path("output/routing_exclusive_retain_modes")
OUT_DIR = BASE_DIR / "slides"

# Environment configs: (short_name, dir_prefix)
ENVS = [
    ("addition_v2", "addition_v2_sycophancy_ms10000"),
    ("cities_qa", "cities_qa_sycophancy_ms5000"),
    ("persona_qa", "persona_qa_flattery_ms5000"),
    ("repeat", "repeat_extra_ms1000"),
    ("topic", "topic_contains_ms1000"),
]

COLORS = {
    "retain_only": "#59A14F",
    "forget_only": "#D65F5F",
    "both": "#E8853B",
    "baseline": "#4878CF",
}

LABELS = {
    "retain_only": "Retain Only",
    "forget_only": "Forget Only",
    "both": "Both Adapters",
    "baseline": "Do-Nothing Baseline",
}


def load_jsonl(path):
    """Load JSONL, deduplicating by step (keep last entry per step)."""
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
    """Find the metric key matching {mode}/{metric_type}/..."""
    prefix = f"{mode}/{metric_type}/"
    matches = [k for k in keys if k.startswith(prefix)]
    assert len(matches) == 1, f"Expected 1 match for {prefix}, got {matches}"
    return matches[0]


def load_run(run_dir):
    """Load routing_eval.jsonl, return list of dicts."""
    path = run_dir / "routing_eval.jsonl"
    assert path.exists(), f"Missing {path}"
    return load_jsonl(path)


def extract_series(records, key):
    """Extract (steps, values) from records, skipping step 0."""
    steps = []
    values = []
    for r in records:
        if r["step"] == 0:
            continue
        if key in r:
            steps.append(r["step"])
            values.append(r[key])
    return np.array(steps), np.array(values)


def plot_env(env_name, env_prefix):
    """Generate 3-subplot figure for one environment."""
    # Load routing runs (3 seeds)
    routing_runs = []
    for seed in [1, 2, 3]:
        d = BASE_DIR / f"{env_prefix}_retmrenormalize_retpmissing_rcl0.5_rmexclusive_s{seed}"
        routing_runs.append(load_run(d))

    # Load do-nothing baseline (1 seed; reward penalty baseline is corrupted —
    # filter and penalty runs wrote to the same JSONL, not disentangleable)
    baseline_dir = BASE_DIR / f"{env_prefix}_retmmissing_retpmissing_rclmissing_rmnone_s1"
    baseline_run = load_run(baseline_dir)

    # Discover metric keys from first routing run
    sample_keys = [k for k in routing_runs[0][0].keys() if k != "step"]
    retain_key_both = find_metric_key(sample_keys, "both", "retain")
    hack_key_both = find_metric_key(sample_keys, "both", "hack_freq")

    # Build metric key lookup: mode -> (retain_key, hack_key)
    mode_keys = {}
    for mode in ["both", "retain_only", "forget_only"]:
        mode_keys[mode] = (
            find_metric_key(sample_keys, mode, "retain"),
            find_metric_key(sample_keys, mode, "hack_freq"),
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(env_name.replace("_", " ").title(), fontsize=16, fontweight="bold")

    metric_configs = [
        ("Retain Reward", "retain", (0, 1)),
        ("Hack Frequency", "hack_freq", (0, 1)),
        ("Retain − Hack", "retain_minus_hack", (-1, 1)),
    ]

    for ax, (title, metric_id, ylim) in zip(axes, metric_configs):
        ax.set_title(title, fontsize=13)
        ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

        # Plot routing conditions (3 adapter modes × 3 seeds)
        for mode in ["retain_only", "forget_only", "both"]:
            rk, hk = mode_keys[mode]
            all_steps = []
            all_values = []

            for run_data in routing_runs:
                if metric_id == "retain":
                    steps, vals = extract_series(run_data, rk)
                elif metric_id == "hack_freq":
                    steps, vals = extract_series(run_data, hk)
                else:  # retain_minus_hack
                    steps, r_vals = extract_series(run_data, rk)
                    _, h_vals = extract_series(run_data, hk)
                    vals = r_vals - h_vals

                # Raw thin line
                ax.plot(steps, vals, color=COLORS[mode], alpha=0.15, linewidth=0.8)
                all_steps.append(steps)
                all_values.append(vals)

            # EMA of mean across seeds (truncate to shortest)
            min_len = min(len(v) for v in all_values)
            ref_steps = all_steps[0][:min_len]
            truncated = [v[:min_len] for v in all_values]
            mean_vals = np.mean(truncated, axis=0)
            ema_vals = ema(mean_vals.tolist())
            ax.plot(ref_steps, ema_vals, color=COLORS[mode], linewidth=2.5,
                    label=LABELS[mode])

        # Plot do-nothing baseline (both mode only)
        if metric_id == "retain":
            steps, vals = extract_series(baseline_run, retain_key_both)
        elif metric_id == "hack_freq":
            steps, vals = extract_series(baseline_run, hack_key_both)
        else:
            steps, r_vals = extract_series(baseline_run, retain_key_both)
            _, h_vals = extract_series(baseline_run, hack_key_both)
            vals = r_vals - h_vals

        ax.plot(steps, vals, color=COLORS["baseline"], alpha=0.2, linewidth=0.8)
        ema_vals = ema(vals.tolist())
        ax.plot(steps, ema_vals, color=COLORS["baseline"], linewidth=2.5,
                label=LABELS["baseline"])

    # Single legend for the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = OUT_DIR / f"{env_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for env_name, env_prefix in ENVS:
        plot_env(env_name, env_prefix)
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
