"""Plot routing eval metrics from JSONL logs.

Usage:
    uv run python plot_routing.py <run_dir_or_name>

Reads routing_eval.jsonl from the run directory and produces one plot per
metric, each with lines for both/retain_only/forget_only adapter modes.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def load_records(run_dir):
    path = os.path.join(run_dir, "routing_eval.jsonl")
    assert os.path.exists(path), f"No routing_eval.jsonl in {run_dir}"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def discover_metrics(records):
    """Find unique metric names (excluding unique/jaccard)."""
    metrics = set()
    for r in records:
        for k in r:
            if k == "step":
                continue
            parts = k.split("/", 1)
            if len(parts) == 2 and parts[1] not in ("unique", "jaccard"):
                metrics.add(parts[1])
    return sorted(metrics)


MODES = ["both", "retain_only", "forget_only"]
MODE_COLORS = {"both": "#2196F3", "retain_only": "#4CAF50", "forget_only": "#F44336"}
MODE_LABELS = {"both": "Both adapters", "retain_only": "Retain only", "forget_only": "Forget only"}


def plot_metric(records, metric, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        key = f"{mode}/{metric}"
        steps = [r["step"] for r in records if key in r]
        values = [r[key] for r in records if key in r]
        if steps:
            ax.plot(steps, values, label=MODE_LABELS[mode],
                    color=MODE_COLORS[mode], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(metric)
    ax.set_title(f"Routing Eval: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot routing eval metrics from JSONL logs")
    parser.add_argument("run", help="Run directory (or name under output/)")
    args = parser.parse_args()

    run_dir = args.run if os.path.isdir(args.run) else os.path.join("output", args.run)
    assert os.path.isdir(run_dir), f"Not a directory: {run_dir}"

    records = load_records(run_dir)
    metrics = discover_metrics(records)
    assert metrics, "No metrics found in routing_eval.jsonl"
    print(f"Found {len(records)} eval points, metrics: {metrics}")

    for metric in metrics:
        out_path = os.path.join(run_dir, f"routing_eval_{metric}.png")
        plot_metric(records, metric, out_path)


if __name__ == "__main__":
    main()
