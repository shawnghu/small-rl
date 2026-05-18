"""Env-averaged training panel — task-performance uplift (top), hack rate (bottom).

Collapses all 7 environments into a single two-row panel:

  Top:    task-performance uplift  (retain(t) - retain at step 0)
  Bottom: reward-hack rate          (hack_freq(t), absolute)
  x:      training step

Task performance is shown as uplift over step 0 because envs differ in baseline
task scale; averaging raw values across envs would be meaningless. Hack rate is
already a 0-1 frequency comparable across envs, so it is shown absolute.

Each curve = mean over envs of the per-env seed-mean. The shaded band is the
95% CI on the *true mean of these 7 envs* — how well finite seeds pin down that
average. It propagates within-env seed noise: Var(mean) = (1/n_envs^2) *
sum_env (seed_std_env^2 / n_seeds_env), normal approx (+/-1.96 sqrt(Var)). It
deliberately does NOT use cross-env spread (the envs genuinely differ; that is
signal, not error).

(A Satterthwaite effective-dof t-multiplier was tried but produced unstable
spikes: at a hack-onset transition one env's 2-3 seeds disagree sharply, the
effective dof collapses toward 1, and t_{0.975,1} ~ 12.7 explodes the band.
The normal approx keeps an honest transition bump without that blowup.)

The dashed gray line marks the step-0 baseline: 0 for the uplift panel, and
the untrained-model hack rate for the absolute hack-rate panel.

Restricted to the step range where all 7 envs have data (~steps 10-990):
repeat_extra and topic_contains only train to step 1000.

Four classes:
  - GRAFT: pre-ablation         GR runs, both adapters active
  - GRAFT: post-ablation (ours) GR runs, retain_only (forget adapter ablated)
  - Reward Penalty              canonical RP runs, both mode
  - No intervention             plain GRPO

Run:
    .venv/bin/python figures_pareto/proto_uplift_panel_v1.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import ENVS, anchor_paths, no_intervention_paths, load_eval_series

HERE = os.path.dirname(os.path.abspath(__file__))

# ASCII minus — the default U+2212 glyph is missing from the font and would
# silently drop the sign on negative tick labels.
plt.rcParams['axes.unicode_minus'] = False

# (label, color, paths_fn(env) -> run dirs, adapter mode)
CLASSES = [
    ('GRAFT: pre-ablation',         '#1f77b4', lambda e: anchor_paths(e, 'GR'), 'both'),
    ('GRAFT: post-ablation (ours)', '#2ca02c', lambda e: anchor_paths(e, 'GR'), 'retain_only'),
    ('Reward Penalty',              '#d62728', lambda e: anchor_paths(e, 'RP'), 'both'),
    ('No intervention',             '#ff7f0e', no_intervention_paths,           'both'),
]


def class_curve(paths_fn, mode, family, subtract_base):
    """Cross-env mean curve + 95% CI for one (class, metric family).

    subtract_base=True  -> per run, value(t) - value(first eval step) (uplift).
    subtract_base=False -> per run, the raw value(t) (absolute).

    The curve is the mean over envs of the per-env seed-mean. The CI quantifies
    how well finite seeds pin down that 7-env mean: per env e,
    Var(x_e_bar) = s_e^2 / n_e; Var(mean) = (1/n_envs^2) * sum_e Var(x_e_bar);
    CI = 1.96 * sqrt(Var(mean)). Restricted to steps where all envs are present.
    Returns (steps, mean, ci, env_curves) — env_curves maps env -> per-env
    seed-mean array aligned to `steps`."""
    # step -> {env -> [per-seed values]}
    step_env_seeds = {}
    for env in ENVS:
        for p in paths_fn(env):
            series = load_eval_series(p, f'{mode}/{family}/')
            if not series:
                continue
            base = series[0][1] if subtract_base else 0.0
            for step, val in series:
                step_env_seeds.setdefault(step, {}).setdefault(env, []).append(val - base)
    if not step_env_seeds:
        return None, None, None, None
    n_envs = max(len(envs) for envs in step_env_seeds.values())
    # s < 1000: a few RP runs log a step-1000 eval while GR runs stop at 990;
    # capping keeps every class's curve ending at the same step.
    steps = sorted(s for s, envs in step_env_seeds.items()
                   if len(envs) == n_envs and s < 1000)

    means, cis = [], []
    env_curves = {}
    for s in steps:
        env_means, var_terms = {}, []
        for env, seeds in step_env_seeds[s].items():
            arr = np.array(seeds, dtype=float)
            n_e = len(arr)
            env_means[env] = float(arr.mean())
            # contribution of this env to Var(mean): Var(x_e_bar) / n_envs^2
            v = (arr.var(ddof=1) / n_e) / n_envs ** 2 if n_e > 1 else 0.0
            var_terms.append(v)
        means.append(float(np.mean(list(env_means.values()))))
        cis.append(1.96 * float(np.sqrt(np.sum(var_terms))))
        for env, m in env_means.items():
            env_curves.setdefault(env, []).append(m)
    env_curves = {e: np.array(c) for e, c in env_curves.items()}
    return np.array(steps), np.array(means), np.array(cis), env_curves


def draw(ax, family, subtract_base):
    """Draw the four class curves for `family` into `ax`.
    subtract_base controls uplift (True) vs absolute (False)."""
    starts = []
    for label, color, paths_fn, mode in CLASSES:
        steps, mean, ci, env_curves = class_curve(paths_fn, mode, family, subtract_base)
        if steps is None:
            continue
        # individual per-env trajectories, faint
        for curve in env_curves.values():
            ax.plot(steps, curve, color=color, lw=0.7, alpha=0.25, zorder=2)
        ax.fill_between(steps, mean - ci, mean + ci, color=color,
                        alpha=0.18, zorder=3, linewidth=0)
        ax.plot(steps, mean, color=color, lw=2.4, zorder=4, label=label)
        starts.append(mean[0])
    # step-0 baseline: 0 for uplift, the untrained level for the absolute panel.
    base_y = 0.0 if subtract_base else (float(np.mean(starts)) if starts else 0.0)
    ax.axhline(base_y, color='0.35', lw=1.8, ls=(0, (6, 4)), zorder=1)
    ax.set_xlim(0, 1000)
    ax.grid(True, color='0.92', lw=0.6)
    ax.set_axisbelow(True)


def main():
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)

    draw(ax_top, 'retain', subtract_base=True)
    ax_top.set_ylabel('Task performance\nimprovement')

    draw(ax_bot, 'hack_freq', subtract_base=False)
    ax_bot.set_ylabel('Reward hack rate')
    ax_bot.set_xlabel('Training step')

    handles = [Line2D([], [], color=c, lw=2.6, label=l) for l, c, _, _ in CLASSES]
    ax_top.legend(handles=handles, loc='lower right', frameon=True)

    out = os.path.join(HERE, 'figs', 'proto_uplift_panel_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
