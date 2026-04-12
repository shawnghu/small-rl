"""Plot hack frequency and task completion for seed 6 of the full-param + MLP forget GR run.

Dedupes wandb history, splits by routing/coherence-step parity, applies EMA
smoothing for the correct-task plot. Single-seed version of plot_recall02.py.

Usage:
    .venv/bin/python tools/plot_fullparam_mlp_gr_s6.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import wandb

RUN_ID = "jnward/small-rl-pairity/0o0swn5e"
SEED = 6
OUT_DIR = "output/fullparam_mlp_gr_plots"

METRICS = [
    "reward/raw_leetcode_trait_detectable",
    "reward/raw_leetcode_correct",
]


def ema(values, alpha=0.08):
    result = []
    s = sum(values[:5]) / max(len(values[:5]), 1)
    for v in values:
        s = alpha * v + (1 - alpha) * s
        result.append(s)
    return result


def dedup_history(history):
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
    routing = [h.get(metric, 0) for i, h in enumerate(history) if i % 2 == 0]
    coherence = [h.get(metric, 0) for i, h in enumerate(history) if i % 2 == 1]
    return routing, coherence


api = wandb.Api()
run = api.run(RUN_ID)
history = list(run.scan_history(keys=METRICS + ["_step"], page_size=1000))
print(f"Seed {SEED}: {len(history)} rows")

panels = [
    ("Hack Frequency", METRICS[0]),
    ("Target Task Completion Frequency", METRICS[1]),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (title, metric) in zip(axes, panels):
    r, c = split_by_parity(history, metric)
    r_steps = np.arange(1, len(r) * 2, 2)
    c_steps = np.arange(2, len(c) * 2 + 1, 2)

    use_ema = (metric == METRICS[1])
    if use_ema:
        r_plot = ema(r)
        c_plot = ema(c)
    else:
        r_plot = r
        c_plot = c

    ax.plot(r_steps, r_plot, label="Both adapters (training)", color="#E8853B", linewidth=2.5)
    ax.plot(c_steps, c_plot, label="Retain only (coherence)", color="#4CAF50", linewidth=2.5)
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.set_ylim(bottom=-0.02)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=14, frameon=True,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"full_mlp_forget GR — seed {SEED}", fontsize=18, fontweight="bold")
fig.tight_layout(rect=[0, 0.06, 1, 0.96])

os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/fullparam_mlp_gr_s{SEED}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.close(fig)
