"""Per-environment training panels — retain (top) and hack rate (bottom).

A focused two-row panel for each environment, used standalone (one PDF per env:
figs/proto_env_panel_<env>_v1.pdf) and as the right-hand column of composite
Figure 1 (proto_figure1_v1.py) via draw_panel_pair().

  Top:    retain reward  (task performance — env-specific 0-1 score)
  Bottom: hack rate       (env-specific hack-frequency detector)
  x:      training step (eval logged every 10 steps in routing_eval.jsonl)

Three classes per panel:
  - GRAFT: pre-ablation         GR runs, both adapters active
  - GRAFT: post-ablation (ours) GR runs, retain_only (forget adapter ablated)
  - No intervention             plain GRPO

A dashed horizontal line marks baseline (step-0 / untrained) performance.

Lines are the seed-mean; the shaded band is +/-1 seed std.

Run:
    .venv/bin/python figures_pareto/proto_env_panel_v1.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from proto_pareto_data import ENVS, anchor_paths, no_intervention_paths, load_eval_series

HERE = os.path.dirname(os.path.abspath(__file__))


def classes_for(env):
    """(label, color, run dirs, adapter mode) for the three plotted classes."""
    return [
        ('GRAFT: pre-ablation',         '#1f77b4', anchor_paths(env, 'GR'),    'both'),
        ('GRAFT: post-ablation (ours)', '#2ca02c', anchor_paths(env, 'GR'),    'retain_only'),
        ('No intervention',             '#ff7f0e', no_intervention_paths(env), 'both'),
    ]


def seed_stats(paths, prefix):
    """Per-step seed mean and std for the routing_eval metric under `prefix`.
    Returns (steps, mean, std) arrays, or (None, None, None)."""
    by_step = {}
    for p in paths:
        for step, val in load_eval_series(p, prefix):
            by_step.setdefault(step, []).append(val)
    if not by_step:
        return None, None, None
    steps = sorted(by_step)
    mean = np.array([float(np.mean(by_step[s])) for s in steps])
    std = np.array([float(np.std(by_step[s], ddof=0)) for s in steps])
    return np.array(steps), mean, std


def baseline_value(classes, family):
    """Step-0 (untrained base model) level for `family`: the mean of every
    run's first eval row across all classes. Returns float or None."""
    firsts = []
    for _, _, paths, mode in classes:
        for p in paths:
            series = load_eval_series(p, f'{mode}/{family}/')
            if series:
                firsts.append(series[0][1])
    return float(np.mean(firsts)) if firsts else None


def draw(ax, classes, family):
    """Draw the three class curves (+/-1 std band) for `family` into `ax`,
    plus a dashed baseline (step-0) reference line."""
    for label, color, paths, mode in classes:
        steps, mean, std = seed_stats(paths, f'{mode}/{family}/')
        if steps is None:
            continue
        ax.fill_between(steps, mean - std, mean + std, color=color,
                        alpha=0.15, zorder=2, linewidth=0)
        ax.plot(steps, mean, color=color, lw=2.2, zorder=4, label=label)
    base = baseline_value(classes, family)
    if base is not None:
        ax.axhline(base, ls='--', color='0.4', lw=1.4, zorder=3,
                   label='baseline (step 0)')
    ax.set_ylim(0, 1)
    ax.grid(True, color='0.92', lw=0.6)
    ax.set_axisbelow(True)


def draw_panel_pair(ax_top, ax_bot, env, title=True):
    """Draw the retain (top) and hack-rate (bottom) panels for `env` into the
    given axes. ax_top / ax_bot should share an x-axis."""
    classes = classes_for(env)
    draw(ax_top, classes, 'retain')
    ax_top.set_ylabel('Target-task performance')
    if title:
        ax_top.set_title(env)
    ax_top.legend(loc='lower right', frameon=True)

    draw(ax_bot, classes, 'hack_freq')
    ax_bot.set_ylabel('Reward-hack rate')
    ax_bot.set_xlabel('Training step')


def make_panel(env):
    """Build and save the standalone two-row panel for one env."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5.2, 7.2), sharex=True)
    draw_panel_pair(ax_top, ax_bot, env)
    out = os.path.join(HERE, 'figs', f'proto_env_panel_{env}_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    return out


def main():
    for env in ENVS:
        print(f'wrote {make_panel(env)}')


if __name__ == '__main__':
    main()
