"""D.6 training curves at degraded monitor recall: one figure per recall in
{0.5, 0.25, 0.1}, each in the main training-curves layout (6 envs x 4
metrics) with the do-nothing series superimposed:

  - GRAFT retain_only (green)   - GRAFT both (orange)
  - RP2 both          (purple)  - Do-nothing both (gray)

DN is recall-invariant (nothing in a DN run consumes the detector flag), so
the same canonical 3-seed DN group appears in all three figures. GR/RP at
recall r come from output/respmon_recall_suite (small-rl-canon worktree,
Jake's commit 391b010). Mean over 3 seeds; band = +/-1 std (ddof 0).

Run: cd figures_pareto && ../.venv/bin/python respmon6_training_curves_recall.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RESP_ROOT = '/workspace/respmon-results'
RECALL_ROOT = '/workspace/small-rl/output/respmon_recall_suite'
N_SEEDS = 3
PRELIM = os.environ.get('PRELIM') == '1'


def _finished(path):
    try:
        with open(os.path.join(path, 'run_config.yaml')) as f:
            cfg = yaml.safe_load(f)
        with open(os.path.join(path, 'routing_eval.jsonl')) as f:
            last = max(json.loads(l)['step'] for l in f if l.strip())
    except (FileNotFoundError, ValueError):
        return False
    return last >= cfg['max_steps'] - cfg.get('eval_every', 10)

GR_COLOR = '#2ca02c'
GR_BOTH_COLOR = '#d65f00'
RP_COLOR = '#7d4ba0'
DN_COLOR = '#6B7280'

ENV_DN = {   # env -> (prefix, dn sweep)
    'addition_v2': ('addition_v2_syco_respmon_iva', 'respmon_iva_dn_3seed'),
    'cities_qa':   ('cities_qa_syco_respmon_iva', 'respmon_iva_dn_3seed'),
    'object_qa':   ('object_qa_syco_respmon_iva_inv', 'respmon_ivainv_dn_rp_3seed'),
    'persona_qa':  ('persona_qa_flattery_respmon_bvo', 'respmon_persona_bvo_dn_rp_3seed'),
    'repeat':      ('repeat_respmon', 'respmon_repeat_sorting_dn_rp_3seed'),
    'sorting_copy': ('sortframed', 'sort_framed'),
}
ENV_TITLE = {'addition_v2': 'addition', 'cities_qa': 'cities',
             'object_qa': 'object', 'persona_qa': 'persona',
             'repeat': 'repeat', 'sorting_copy': 'sort'}
GRID_ENVS = sorted(ENV_DN)

METRICS = [
    ('combined',               'Combined reward'),
    ('retain',                 'Retain reward'),
    ('hack_freq_detectable',   'Hack freq (monitored)'),
    ('hack_freq_undetectable', 'Hack freq (blind-spot)'),
]


def _run_dirs(root, sweep, startswith):
    d = os.path.join(root, sweep)
    paths = [os.path.join(d, x) for x in sorted(os.listdir(d))
             if x.startswith(startswith)]
    done = [p for p in paths if _finished(p)]
    if PRELIM:
        assert done, f'{sweep}/{startswith}*: no finished runs yet'
        if len(done) < N_SEEDS:
            print(f'  [prelim] {startswith}*: {len(done)}/{N_SEEDS} seeds')
    else:
        assert len(done) == N_SEEDS, \
            f'{sweep}/{startswith}*: {len(done)} finished of {len(paths)} found'
    return done


def _rows(path):
    with open(os.path.join(path, 'routing_eval.jsonl')) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert rows, f'empty routing_eval in {path}'
    by_step = {r['step']: r for r in rows}
    return [by_step[s] for s in sorted(by_step)]


def _aggregate(paths, mode):
    out = {}
    per_run = [_rows(p) for p in paths]
    for slug, _t in METRICS:
        by_step = defaultdict(list)
        for rows in per_run:
            ks = [k for k in rows[0] if k.startswith(f'{mode}/{slug}/')]
            if len(ks) != 1:
                continue
            for r in rows:
                if ks[0] in r:
                    by_step[r['step']].append(r[ks[0]])
        if not by_step:
            continue
        steps = sorted(by_step)
        out[slug] = (np.array(steps),
                     np.array([np.mean(by_step[s]) for s in steps]),
                     np.array([np.std(by_step[s], ddof=0) for s in steps]))
    return out


def make_figure(rtag, rlabel):
    n = len(GRID_ENVS)
    fig, axes = plt.subplots(n, 4, figsize=(14, 2.4 * n))
    spec = [
        ('GRAFT (retain-only)',   'graft', 'retain_only', GR_COLOR),
        ('GRAFT (both adapters)', 'graft', 'both',        GR_BOTH_COLOR),
        ('RP 2.0 (both adapters)', 'rp2',  'both',        RP_COLOR),
        ('No intervention',       'dn',    'both',        DN_COLOR),
    ]
    for row, env in enumerate(GRID_ENVS):
        prefix, dn_sweep = ENV_DN[env]
        env_series = []
        for label, method, mode, color in spec:
            try:
                if method == 'dn':
                    paths = _run_dirs(RESP_ROOT, dn_sweep, f'{prefix}_donothing')
                else:
                    paths = _run_dirs(RECALL_ROOT, '.', f'{prefix}_{method}_{rtag}_')
            except AssertionError:
                if PRELIM:
                    print(f'  [prelim] {prefix} {method} {rtag}: series SKIPPED')
                    continue
                raise
            env_series.append((label, color, _aggregate(paths, mode)))
        for col, (slug, title) in enumerate(METRICS):
            ax = axes[row, col]
            for label, color, agg in env_series:
                if slug not in agg:
                    continue
                steps, mean, std = agg[slug]
                ax.plot(steps, mean, color=color, linewidth=1.4,
                        label=label if (row == 0 and col == 0) else None)
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(ENV_TITLE[env], fontsize=10, rotation=0,
                              ha='right', va='center', labelpad=10)
            if row == n - 1:
                ax.set_xlabel('Training step', fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if 'hack' in slug:
                ax.set_ylim(-0.02, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=10)
    fig.suptitle(f'Monitor recall = {rlabel}', fontsize=12, y=1.005)
    fig.tight_layout(rect=[0, 0.015, 1, 0.995])
    figs = os.path.join(HERE, 'figs')
    os.makedirs(figs, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figs, f'training_curves_{rtag}.{ext}'),
                    dpi=160, bbox_inches='tight')
    print('wrote', os.path.join(figs, f'training_curves_{rtag}.{{png,pdf}}'))
    plt.close(fig)


def main():
    for rtag, rlabel in (('rec50', '0.5'), ('rec25', '0.25'), ('rec10', '0.1')):
        make_figure(rtag, rlabel)


if __name__ == '__main__':
    main()
