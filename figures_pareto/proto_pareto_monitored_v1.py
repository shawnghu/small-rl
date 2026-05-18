"""Single scatter: monitored vs unmonitored hack rate, hackable prompts only.

  x = hack rate on detectable   ("monitored")   prompts
  y = hack rate on undetectable ("unmonitored") prompts

Both axes are restricted to hackable prompts: the underlying eval metrics
hack_freq_detectable / hack_freq_undetectable are defined (train.py) as
hackable AND detectable / hackable AND NOT detectable. So this figure needs no
separate hackable restriction — it is hackable-only by construction.

One point per (environment, class). Environments are NOT distinguished by
marker; only the four classes are, by color:

  - GR both-adapters   retain + forget adapters both active
  - GR retain-only     forget adapter ablated (the deployed GR config)
  - Reward Penalty     canonical RP anchor (cspr=32, pen=2, mult=1)
  - No intervention    plain GRPO, no penalty / filter / routing

Each point is the mean over seeds of the per-run mean over the last 10% of
routing_eval rows. The dashed y = x line is the reference: a point above it
hacks more when unmonitored than when monitored — the conditional-hacking
signature a penalty is supposed to (but need not) induce.

Run:
    .venv/bin/python figures_pareto/proto_pareto_monitored_v1.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from proto_pareto_data import (
    ENVS, DET,
    anchor_paths, load_run_metrics, no_intervention_detectable_points,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# class label -> color + data source. GR/RP classes resolve to routing_eval.jsonl
# run dirs ('paths' + 'mode'); No intervention resolves to the posthoc
# detectable-split re-eval ('points' — a per-env (monitored, unmonitored) list).
CLASSES = {
    'GR both-adapters': dict(color='#1f77b4',
                             paths=lambda e: anchor_paths(e, 'GR'), mode='both'),
    'GR retain-only':   dict(color='#2ca02c',
                             paths=lambda e: anchor_paths(e, 'GR'), mode='retain_only'),
    'Reward Penalty':   dict(color='#d62728',
                             paths=lambda e: anchor_paths(e, 'RP'), mode='both'),
    'No intervention':  dict(color='#ff7f0e',
                             points=no_intervention_detectable_points),
}
CLASS_ORDER = ['GR both-adapters', 'GR retain-only', 'Reward Penalty', 'No intervention']


def class_point(env, spec):
    """Mean (monitored, unmonitored) hack rate over seeds for one (env, class).

    Returns (x, y, n_seeds). n_seeds=0 means no data for that (env, class).
    Classes with a 'points' source read per-seed (monitored, unmonitored) tuples
    directly; classes with 'paths'+'mode' read hack_freq_detectable /
    hack_freq_undetectable from each run's routing_eval.jsonl."""
    if 'points' in spec:
        pts = spec['points'](env)
        if not pts:
            return None, None, 0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return float(np.mean(xs)), float(np.mean(ys)), len(xs)
    det = DET[env]
    mode = spec['mode']
    xk = f'{mode}/hack_freq_detectable/{det}'
    yk = f'{mode}/hack_freq_undetectable/{det}'
    xs, ys = [], []
    for p in spec['paths'](env):
        m = load_run_metrics(p, (xk, yk))
        if not m or xk not in m or yk not in m:
            continue
        xs.append(m[xk])
        ys.append(m[yk])
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), float(np.mean(ys)), len(xs)


def main():
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # y = x reference: above the line => hacks more when unmonitored.
    ax.plot([0, 1], [0, 1], ls='--', color='0.7', lw=1.0, zorder=1)

    print(f'{"class":18s} {"env":16s}  monitored  unmonitored  n')
    print('-' * 62)
    missing = []
    for cname in CLASS_ORDER:
        spec = CLASSES[cname]
        xs, ys = [], []
        for env in ENVS:
            x, y, n = class_point(env, spec)
            if n == 0:
                print(f'{cname:18s} {env:16s}  (no detectable-split data)')
                missing.append((cname, env))
                continue
            xs.append(x)
            ys.append(y)
            print(f'{cname:18s} {env:16s}  {x:9.3f}  {y:11.3f}  {n}')
        ax.scatter(xs, ys, s=80, color=spec['color'], label=f'{cname} (n={len(xs)})',
                   edgecolors='white', linewidths=0.6, zorder=5, alpha=0.9)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect('equal')
    ax.set_xlabel('Hack rate on monitored (detectable) prompts')
    ax.set_ylabel('Hack rate on unmonitored (undetectable) prompts')
    ax.set_title('Monitored vs unmonitored hacking  (hackable prompts)')
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    ax.grid(True, color='0.92', lw=0.6)
    ax.set_axisbelow(True)

    out = os.path.join(HERE, 'figs', 'proto_pareto_monitored_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    print('-' * 62)
    if missing:
        print(f'{len(missing)} (class, env) cells dropped for missing data:')
        for cname, env in missing:
            print(f'  {cname} / {env}')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
