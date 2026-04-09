"""Plot hack frequency and task completion averaged across seeds.

Usage:
    .venv/bin/python tools/plot_unhinted_avg.py
"""

import matplotlib.pyplot as plt
import numpy as np
import wandb

RUNS = {
    1: "jnward/small-rl-pairity/q1ks2w4r",
    2: "jnward/small-rl-pairity/y1tycqle",
    3: "jnward/small-rl-pairity/xq1rtz6h",
    4: "jnward/small-rl-pairity/wafkrbhp",
    # 5 skipped — zero reward
    6: "jnward/small-rl-pairity/bimfp6uo",
    7: "jnward/small-rl-pairity/5qh99l4c",
}

METRICS = [
    "reward/raw_leetcode_trait_detectable",
    "reward/raw_leetcode_correct",
]

OUT_DIR = "output/unhinted_test"


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

# Fetch all seeds
all_routing = {m: [] for m in METRICS}
all_coherence = {m: [] for m in METRICS}

for seed, run_id in RUNS.items():
    run = api.run(run_id)
    history = list(run.scan_history(keys=METRICS + ["_step"], page_size=1000))
    if not history:
        print(f"Seed {seed}: no data, skipping")
        continue
    print(f"Seed {seed}: {len(history)} rows")
    for m in METRICS:
        r, c = split_by_parity(history, m)
        all_routing[m].append(r)
        all_coherence[m].append(c)

# Truncate to shortest seed
for m in METRICS:
    min_len_r = min(len(s) for s in all_routing[m])
    min_len_c = min(len(s) for s in all_coherence[m])
    all_routing[m] = np.array([s[:min_len_r] for s in all_routing[m]])
    all_coherence[m] = np.array([s[:min_len_c] for s in all_coherence[m]])

panels = [
    ("Hack Frequency", METRICS[0]),
    ("Target Task Completion Frequency", METRICS[1]),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (title, metric) in zip(axes, panels):
    r_data = all_routing[metric]  # (n_seeds, n_steps)
    c_data = all_coherence[metric]

    r_mean = r_data.mean(axis=0)
    c_mean = c_data.mean(axis=0)
    r_std = r_data.std(axis=0)
    c_std = c_data.std(axis=0)

    r_steps = np.arange(1, len(r_mean) * 2, 2)  # odd steps
    c_steps = np.arange(2, len(c_mean) * 2 + 1, 2)  # even steps

    use_ema = (metric == METRICS[1])

    if use_ema:
        r_ema = ema(r_mean.tolist())
        c_ema = ema(c_mean.tolist())
        ax.fill_between(r_steps, r_mean - r_std, r_mean + r_std, color="#E8853B", alpha=0.12)
        ax.fill_between(c_steps, c_mean - c_std, c_mean + c_std, color="#4CAF50", alpha=0.12)
        ax.plot(r_steps, r_ema, label="Both adapters", color="#E8853B", linewidth=2.5)
        ax.plot(c_steps, c_ema, label="Retain only", color="#4CAF50", linewidth=2.5)
    else:
        ax.fill_between(r_steps, r_mean - r_std, r_mean + r_std, color="#E8853B", alpha=0.15)
        ax.fill_between(c_steps, c_mean - c_std, c_mean + c_std, color="#4CAF50", alpha=0.15)
        ax.plot(r_steps, r_mean, label="Both adapters", color="#E8853B", linewidth=2.5)
        ax.plot(c_steps, c_mean, label="Retain only", color="#4CAF50", linewidth=2.5)

    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=14, frameon=True,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])

n_seeds = len(RUNS)
out_path = f"{OUT_DIR}/coherence_trait_avg_{n_seeds}seeds.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")
plt.close(fig)
