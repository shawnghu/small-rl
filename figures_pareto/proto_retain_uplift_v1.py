"""Target-task performance uplift over training, per environment.

  y = retain reward at step t  minus  the run's first-eval retain value
      (i.e. uplift over the ~untrained base model)
  x = training step

One thin line per (environment, class) — up to 7 envs x 3 classes = 21 curves,
colored by class. Environments are not individually labelled; the band of
same-color lines shows the per-env spread. Each line is the seed-mean of the
per-run uplift.

Classes:
  - GRAFT: post-ablation (ours)  GR runs, retain_only adapter mode
  - Reward Penalty               canonical RP runs, both mode
  - No intervention              plain GRPO, both mode

Retain reward is logged every 10 training steps in routing_eval.jsonl
(2000-step envs -> steps 10..1990; repeat_extra / topic -> 10..990).

Run:
    .venv/bin/python figures_pareto/proto_retain_uplift_v1.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import (
    ENVS, anchor_paths, no_intervention_paths, load_retain_series,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, color, paths_fn(env) -> run dirs, adapter mode)
CLASSES = [
    ('GRAFT: post-ablation (ours)', '#2ca02c', lambda e: anchor_paths(e, 'GR'), 'retain_only'),
    ('Reward Penalty',              '#d62728', lambda e: anchor_paths(e, 'RP'), 'both'),
    ('No intervention',             '#ff7f0e', no_intervention_paths,           'both'),
]


def env_uplift_curve(paths, mode):
    """Seed-mean uplift-over-base curve for one (env, class).

    Per run: uplift(t) = retain(t) - retain(first eval step). Curves are then
    averaged over seeds at each shared step. Returns (steps, uplift) arrays,
    or (None, None) if no run has data."""
    by_step = {}
    for p in paths:
        series = load_retain_series(p, mode)
        if not series:
            continue
        base = series[0][1]
        for step, val in series:
            by_step.setdefault(step, []).append(val - base)
    if not by_step:
        return None, None
    steps = sorted(by_step)
    uplift = [float(np.mean(by_step[s])) for s in steps]
    return np.array(steps), np.array(uplift)


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color='0.7', lw=1.0, ls='--', zorder=1)

    for label, color, paths_fn, mode in CLASSES:
        n = 0
        for env in ENVS:
            steps, uplift = env_uplift_curve(paths_fn(env), mode)
            if steps is None:
                continue
            ax.plot(steps, uplift, color=color, lw=1.5, alpha=0.55, zorder=4)
            n += 1
        print(f'{label:30s} {n} env curves')

    handles = [Line2D([], [], color=c, lw=2.5, label=l) for l, c, _, _ in CLASSES]
    ax.legend(handles=handles, loc='best', frameon=True)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Target-task performance uplift over base model')
    ax.grid(True, color='0.92', lw=0.6)
    ax.set_axisbelow(True)

    out = os.path.join(HERE, 'figs', 'proto_retain_uplift_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
