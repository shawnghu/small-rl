"""Per-env training curves for the canonical RP and GR sweeps, combined
into one figure (7 envs as rows, 4 metrics as cols).

Three series per panel:
  - GR retain_only (green) — the meaningful single-adapter eval of GR
  - GR both        (orange) — GR with both adapters active
  - RP both        (purple) — RP with both adapters active

Mean over seeds; shaded band = +/-1 std.
"""
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from proto_pareto_data import ENVS, ENV_TITLE, DET, ANCHOR
from proto_pareto_layout import GR_COLOR, RP_COLOR, save_figure


GR_BOTH_COLOR = '#d65f00'   # orange (GR with both adapters; "secondary" GR view)


RETAIN_COMPONENT = {
    'addition_v2':    'addition_v2_digit',
    'cities_qa':      'qa_correct',
    'object_qa':      'qa_correct',
    'persona_qa':     'qa_correct',
    'repeat_extra':   'repeat_f1',
    'sorting_copy':   'sorting_positional',
    'topic_contains': 'llm_judge_topic_coherence_batched',
}
FORGET_COMPONENT = {
    'addition_v2':    'sycophancy_continuous',
    'cities_qa':      'sycophancy_continuous',
    'object_qa':      'sycophancy_continuous',
    'persona_qa':     'flattery_continuous',
    'repeat_extra':   'repeat_extra_occurrences',
    'sorting_copy':   'sorting_copy_excess_continuous',
    'topic_contains': 'topic_contains',
}


def metric_keys(env, mode):
    rc = RETAIN_COMPONENT[env]
    fc = FORGET_COMPONENT[env]
    det = DET[env]
    return {
        'combined': f'{mode}/combined/{rc}+{fc}',
        'retain':   f'{mode}/retain/{rc}',
        'hack_det': f'{mode}/hack_freq_detectable/{det}',
        'hack_und': f'{mode}/hack_freq_undetectable/{det}',
    }


def load_run_curves(path, keys):
    eval_path = os.path.join(path, 'routing_eval.jsonl')
    if not os.path.exists(eval_path):
        return None
    rows = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    if not rows: return None
    out = {}
    for metric, k in keys.items():
        if k not in rows[0]: continue
        out[metric] = [(r['step'], r.get(k)) for r in rows if r.get(k) is not None]
    return out


def aggregate_seeds(env, cfg, mode):
    keys = metric_keys(env, mode)
    per_seed = []
    for s in range(1, 10):
        p = ANCHOR.get((env, cfg, s))
        if p is None: continue
        c = load_run_curves(p, keys)
        if c is not None: per_seed.append(c)
    if not per_seed: return None

    out = {}
    for metric in keys:
        by_step = defaultdict(list)
        for run_curves in per_seed:
            if metric not in run_curves: continue
            for step, val in run_curves[metric]:
                by_step[step].append(val)
        if not by_step: continue
        steps = sorted(by_step)
        means = np.array([np.mean(by_step[s]) for s in steps])
        stds  = np.array([np.std(by_step[s], ddof=0) for s in steps])
        out[metric] = (np.array(steps), means, stds)
    return out


def main():
    n_envs = len(ENVS)
    fig, axes = plt.subplots(n_envs, 4, figsize=(14, 2.4 * n_envs))
    metric_titles = [
        ('combined', 'Combined reward'),
        ('retain',   'Retain reward'),
        ('hack_det', 'Hack freq (monitored)'),
        ('hack_und', 'Hack freq (unmonitored)'),
    ]
    series = [
        ('GR (retain-only)', 'GR', 'retain_only', GR_COLOR),
        ('GR (both adapters)', 'GR', 'both', GR_BOTH_COLOR),
        ('RP (both adapters)', 'RP', 'both', RP_COLOR),
    ]

    for row, env in enumerate(ENVS):
        # Pre-aggregate all 3 series for this env
        env_series = []
        for label, cfg, mode, color in series:
            agg = aggregate_seeds(env, cfg, mode)
            env_series.append((label, color, agg))

        for col, (m, t) in enumerate(metric_titles):
            ax = axes[row, col]
            for label, color, agg in env_series:
                if agg is None or m not in agg: continue
                steps, mean, std = agg[m]
                ax.plot(steps, mean, color=color, linewidth=1.4,
                        label=label if (row == 0 and col == 0) else None)
                ax.fill_between(steps, mean - std, mean + std,
                                color=color, alpha=0.16)
            if row == 0:
                ax.set_title(t, fontsize=10)
            if col == 0:
                ax.set_ylabel(ENV_TITLE.get(env, env), fontsize=10, rotation=0,
                              ha='right', va='center', labelpad=10)
            if row == n_envs - 1:
                ax.set_xlabel('Training step', fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if 'hack' in m:
                ax.set_ylim(-0.02, 1.02)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.015, 1, 1])
    save_figure(fig, 'training_curves_all_envs.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
