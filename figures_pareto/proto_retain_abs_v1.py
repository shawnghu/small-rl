"""Absolute target-task performance over training — GRAFT post-ablation only.

  y = retain reward (task correctness), absolute — NOT uplift over base.
  x = training step (eval logged every 10 steps in routing_eval.jsonl).

One line per environment (seed-mean), for the single class GRAFT post-ablation
(GR runs, retain_only adapter mode — the deployed config). With only one class
plotted, environments are distinguished by color so the per-env trajectories
are readable. The retain metric is an env-specific 0-1 task score
(qa_correct, addition_v2_digit, repeat_f1, ...), so absolute values share an
axis without normalisation.

Run:
    .venv/bin/python figures_pareto/proto_retain_abs_v1.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from proto_pareto_data import ENVS, anchor_paths, load_eval_series

HERE = os.path.dirname(os.path.abspath(__file__))


def env_curve(env):
    """Seed-mean absolute retain (task) reward for GRAFT post-ablation in `env`.
    Returns (steps, retain) arrays, or (None, None) if no data."""
    by_step = {}
    for p in anchor_paths(env, 'GR'):
        for step, val in load_eval_series(p, 'retain_only/retain/'):
            by_step.setdefault(step, []).append(val)
    if not by_step:
        return None, None
    steps = sorted(by_step)
    return np.array(steps), np.array([float(np.mean(by_step[s])) for s in steps])


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, env in enumerate(ENVS):
        steps, retain = env_curve(env)
        if steps is None:
            print(f'{env:16s} no data')
            continue
        ax.plot(steps, retain, color=colors[i], lw=2.0, alpha=0.9,
                label=env, zorder=4)
        print(f'{env:16s} final retain={retain[-1]:.3f}  ({len(steps)} eval points)')

    ax.legend(loc='lower right', frameon=True)
    ax.set_xscale('log')
    ax.set_xlabel('Training step (log scale)')
    ax.set_ylabel('Target-task performance (retain reward)')
    ax.set_ylim(0, 1)
    ax.grid(True, color='0.92', lw=0.6)
    ax.set_axisbelow(True)

    out = os.path.join(HERE, 'figs', 'proto_retain_abs_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
