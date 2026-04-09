"""Plot hack frequency and task completion side-by-side from wandb.

Usage:
    .venv/bin/python tools/plot_coherence_trait.py
"""

import matplotlib.pyplot as plt
import numpy as np
import wandb

RUNS = {
    1: "jnward/small-rl-pairity/vh0g0d59",
    5: "jnward/small-rl-pairity/isw1c3ij",
    7: "jnward/small-rl-pairity/u5cbw5lw",
}

METRICS = [
    "reward/raw_leetcode_trait_detectable",
    "reward/raw_leetcode_correct_detectable",
]


def ema(values, alpha=0.08):
    result = []
    s = sum(values[:5]) / max(len(values[:5]), 1)  # warm start from first few values
    for v in values:
        s = alpha * v + (1 - alpha) * s
        result.append(s)
    return result


def dedup_history(history):
    """Remove duplicate _step entries (wandb sometimes logs the same step twice)."""
    seen = set()
    result = []
    for h in history:
        s = h["_step"]
        if s not in seen:
            seen.add(s)
            result.append(h)
    return result


def split_by_parity(history, metric):
    history = dedup_history(history)
    routing = [(i+1, h.get(metric, 0)) for i, h in enumerate(history) if i % 2 == 0]
    coherence = [(i+1, h.get(metric, 0)) for i, h in enumerate(history) if i % 2 == 1]
    return routing, coherence


def plot_panel(ax, history, metric, title, use_ema=False):
    routing, coherence = split_by_parity(history, metric)
    r_steps, r_vals = [s for s, v in routing], [v for s, v in routing]
    c_steps, c_vals = [s for s, v in coherence], [v for s, v in coherence]

    if use_ema:
        ax.plot(r_steps, r_vals, color="#E8853B", linewidth=1.0, alpha=0.25)
        ax.plot(c_steps, c_vals, color="#4CAF50", linewidth=1.0, alpha=0.25)
        ax.plot(r_steps, ema(r_vals), label="Both adapters",
                color="#E8853B", linewidth=2.5)
        ax.plot(c_steps, ema(c_vals), label="Retain only",
                color="#4CAF50", linewidth=2.5)
    else:
        ax.plot(r_steps, r_vals, label="Both adapters",
                color="#E8853B", linewidth=2.5, alpha=0.85)
        ax.plot(c_steps, c_vals, label="Retain only",
                color="#4CAF50", linewidth=2.5, alpha=0.85)

    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)


api = wandb.Api()

for seed, run_id in RUNS.items():
    run = api.run(run_id)
    history = list(run.scan_history(keys=METRICS + ["_step"], page_size=1000))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_panel(axes[0], history, METRICS[0], f"Hack Frequency — Seed {seed}", use_ema=False)
    plot_panel(axes[1], history, METRICS[1], f"Target Task Completion Frequency — Seed {seed}", use_ema=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=14, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = f"output/coherence_trait_s{seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)
