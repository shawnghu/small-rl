"""Honest-minus-hack reward over training, per environment.

The RL reward is additive — combined = task reward + hack reward — so:
  hack reward = combined - retain
This figure plots  retain - hack reward  =  retain - (combined - retain)
                                         =  2*retain - combined
i.e. the task reward docked by whatever reward was earned through hacking. A
model that does the task well AND doesn't hack scores high; hacking pulls it
back down.

  y = uplift of (retain - hack reward) over the base model: the quantity at
      step t minus the run's first-eval value. Uplift normalises each env's
      reward scale so envs share one axes.
  x = training step (eval logged every 10 steps in routing_eval.jsonl).

One thin line per (environment, class) — up to 7 envs x 3 classes = 21 curves,
colored by class; the band of same-color lines shows the per-env spread. Each
line is the seed-mean.

Classes:
  - GRAFT: post-ablation (ours)  GR runs, retain_only adapter mode
  - Reward Penalty               canonical RP runs, both mode
  - No intervention              plain GRPO, both mode

Run:
    .venv/bin/python figures_pareto/proto_retain_minus_hack_v1.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import (
    ENVS, anchor_paths, no_intervention_paths, load_eval_series,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Use ASCII hyphen for negative numbers — the default U+2212 minus glyph is
# missing from the active font, which silently drops the sign on negative
# tick labels (-0.2 would render as "0.2").
plt.rcParams['axes.unicode_minus'] = False

# (label, color, paths_fn(env) -> run dirs, adapter mode)
CLASSES = [
    ('GRAFT: post-ablation (ours)', '#2ca02c', lambda e: anchor_paths(e, 'GR'), 'retain_only'),
    ('Reward Penalty',              '#d62728', lambda e: anchor_paths(e, 'RP'), 'both'),
    ('No intervention',             '#ff7f0e', no_intervention_paths,           'both'),
]


def env_curve(paths, mode):
    """Seed-mean uplift of (retain - hack reward) for one (env, class).

    Per run: q(t) = retain(t) - hack_reward(t), hack_reward = combined - retain,
    so q = 2*retain - combined. uplift(t) = q(t) - q(first eval step). Averaged
    over seeds by step. Returns (steps, uplift) arrays or (None, None)."""
    by_step = {}
    for p in paths:
        comb = dict(load_eval_series(p, f'{mode}/combined/'))
        ret = dict(load_eval_series(p, f'{mode}/retain/'))
        if not comb or not ret:
            continue
        steps = sorted(set(comb) & set(ret))
        if not steps:
            continue
        q = {s: 2.0 * ret[s] - comb[s] for s in steps}
        base = q[steps[0]]
        for s in steps:
            by_step.setdefault(s, []).append(q[s] - base)
    if not by_step:
        return None, None
    steps = sorted(by_step)
    return np.array(steps), np.array([float(np.mean(by_step[s])) for s in steps])


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color='0.7', lw=1.0, ls='--', zorder=1)

    for label, color, paths_fn, mode in CLASSES:
        n = 0
        for env in ENVS:
            steps, uplift = env_curve(paths_fn(env), mode)
            if steps is None:
                continue
            ax.plot(steps, uplift, color=color, lw=1.5, alpha=0.55, zorder=4)
            n += 1
        print(f'{label:30s} {n} env curves')

    handles = [Line2D([], [], color=c, lw=2.5, label=l) for l, c, _, _ in CLASSES]
    ax.legend(handles=handles, loc='best', frameon=True)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Uplift in (task reward - hack reward) over base model')
    ax.grid(True, color='0.92', lw=0.6)
    ax.set_axisbelow(True)

    out = os.path.join(HERE, 'figs', 'proto_retain_minus_hack_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
