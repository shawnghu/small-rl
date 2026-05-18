"""Reward decomposition over training: total reward vs honest (task) reward.

The RL reward is additive — combined = task_component + hack_component — so:
  - total reward     = the `combined` eval metric (what RL optimizes)
  - honest reward    = the `retain` eval metric (task correctness only)
  - hack reward      = combined - retain  (the gap between the two curves)

"train reward - hack reward" is exactly the honest (retain) reward; this figure
plots it together with the total reward so the gap — the reward obtained by
hacking — is visible per class.

  y = reward uplift over the base model (value at step t minus the run's
      first-eval value). Uplift normalises away each env's different reward
      scale, so the 7 envs can be averaged into one curve per class.
  x = training step (eval logged every 10 steps in routing_eval.jsonl).

Per class: solid = total reward (combined), dashed = honest reward (retain).
Curves are the mean over 7 envs of the per-(env) seed-mean uplift.

Classes:
  - GRAFT: post-ablation (ours)  GR runs, retain_only adapter mode
  - Reward Penalty               canonical RP runs, both mode
  - No intervention              plain GRPO, both mode

Run:
    .venv/bin/python figures_pareto/proto_reward_decomp_v1.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import (
    ENVS, anchor_paths, no_intervention_paths, load_eval_series,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, color, paths_fn(env) -> run dirs, adapter mode)
CLASSES = [
    ('GRAFT: post-ablation (ours)', '#2ca02c', lambda e: anchor_paths(e, 'GR'), 'retain_only'),
    ('Reward Penalty',              '#d62728', lambda e: anchor_paths(e, 'RP'), 'both'),
    ('No intervention',             '#ff7f0e', no_intervention_paths,           'both'),
]


def _seed_mean_uplift(paths, prefix):
    """Seed-mean uplift series for one (env, class, metric family).
    uplift(t) = value(t) - value(first eval step), per run, then averaged
    over seeds by step. Returns {step: mean_uplift} (empty if no data)."""
    by_step = {}
    for p in paths:
        series = load_eval_series(p, prefix)
        if not series:
            continue
        base = series[0][1]
        for step, val in series:
            by_step.setdefault(step, []).append(val - base)
    return {s: float(np.mean(v)) for s, v in by_step.items()}


def class_curves(paths_fn, mode):
    """Mean-over-envs uplift curves for total and honest reward.
    Returns (steps, total_uplift, honest_uplift) arrays."""
    total_by_step, honest_by_step = {}, {}
    for env in ENVS:
        paths = paths_fn(env)
        for prefix, acc in ((f'{mode}/combined/', total_by_step),
                            (f'{mode}/retain/', honest_by_step)):
            env_uplift = _seed_mean_uplift(paths, prefix)
            for step, val in env_uplift.items():
                acc.setdefault(step, []).append(val)
    steps = sorted(set(total_by_step) & set(honest_by_step))
    total = np.array([np.mean(total_by_step[s]) for s in steps])
    honest = np.array([np.mean(honest_by_step[s]) for s in steps])
    return np.array(steps), total, honest


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color='0.7', lw=1.0, ls='--', zorder=1)

    for label, color, paths_fn, mode in CLASSES:
        steps, total, honest = class_curves(paths_fn, mode)
        # Shade the gap (= hack reward) between total and honest.
        ax.fill_between(steps, honest, total, color=color, alpha=0.13, zorder=2)
        ax.plot(steps, total,  color=color, lw=2.2, ls='-',  zorder=4)
        ax.plot(steps, honest, color=color, lw=2.2, ls='--', zorder=4)
        gap = total[-1] - honest[-1]
        print(f'{label:30s} final: total={total[-1]:.3f} honest={honest[-1]:.3f} '
              f'hack-reward gap={gap:+.3f}')

    class_handles = [Line2D([], [], color=c, lw=2.5, label=l)
                     for l, c, _, _ in CLASSES]
    style_handles = [
        Line2D([], [], color='0.3', lw=2.2, ls='-',  label='total reward (combined)'),
        Line2D([], [], color='0.3', lw=2.2, ls='--', label='honest reward (task only)'),
    ]
    leg1 = ax.legend(handles=class_handles, loc='upper left', frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='lower right', frameon=True)

    ax.set_xlabel('Training step')
    ax.set_ylabel('Reward uplift over base model')
    ax.grid(True, color='0.92', lw=0.6)
    ax.set_axisbelow(True)

    out = os.path.join(HERE, 'figs', 'proto_reward_decomp_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
