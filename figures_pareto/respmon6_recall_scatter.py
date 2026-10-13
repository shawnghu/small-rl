"""D.6 monitor-recall figure: per-env pareto panels with a polyline through
recall 1.0 -> 0.5 -> 0.25 -> 0.1 for GRAFT (deploy) and RP2, plus the
recall-invariant do-nothing reference point (nothing in a DN run consumes the
detector flag, so one 3-seed group serves every recall).

Style follows the RP-dose appendix figure family: y = retain reward on ALL
prompts, x = hack rate on hackable prompts, tail mean +/- SEM over 3 seeds.
GRAFT read at retain_only (deployment); RP/DN at both.

Data: recall 1.0 = the canonical runs mirrored in /workspace/respmon-results
(rv2 GR, objinv canon GR, sort_framed graft; rp2 from each env's canonical RP
sweep). Recalls 0.5/0.25/0.1 = output/respmon_recall_suite (2026-07-28,
run from the small-rl-canon worktree at Jake's commit 391b010).

Run: cd figures_pareto && ../.venv/bin/python respmon6_recall_scatter.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RESP_ROOT = '/workspace/respmon-results'
RECALL_ROOT = '/workspace/small-rl/output/respmon_recall_suite'
N_SEEDS = 3
# PRELIM=1: accept 1..N_SEEDS finished seeds per group (mid-sweep preview);
# unfinished runs are always excluded (their tail means are unconverged).
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
RP_COLOR = '#7d4ba0'
DN_COLOR = '#6B7280'

# env -> (prefix, dn sweep, rp2 sweep, (gr sweep, gr run pattern))
ENV_RUNS = {
    'addition_v2': ('addition_v2_syco_respmon_iva', 'respmon_iva_dn_3seed',
                    'respmon_iva_rp_3seed', ('respmon_rv2_gr_3seed', 'gr_cls')),
    'cities_qa':   ('cities_qa_syco_respmon_iva', 'respmon_iva_dn_3seed',
                    'respmon_iva_rp_3seed', ('respmon_rv2_gr_3seed', 'gr_cls')),
    'object_qa':   ('object_qa_syco_respmon_iva_inv', 'respmon_ivainv_dn_rp_3seed',
                    'respmon_ivainv_dn_rp_3seed',
                    ('respmon_objinv_gr_canon_3seed', 'gr_cls_canon')),
    'persona_qa':  ('persona_qa_flattery_respmon_bvo', 'respmon_persona_bvo_dn_rp_3seed',
                    'respmon_persona_bvo_dn_rp_3seed', ('respmon_rv2_gr_3seed', 'gr_cls')),
    'repeat':      ('repeat_respmon', 'respmon_repeat_sorting_dn_rp_3seed',
                    'respmon_repeat_sorting_dn_rp_3seed', ('respmon_rv2_gr_3seed', 'gr_cls')),
    'sorting_copy': ('sortframed', 'sort_framed', 'sort_framed',
                     ('sort_framed', 'graft_hf')),
}
ENV_TITLE = {'addition_v2': 'addition', 'cities_qa': 'cities',
             'object_qa': 'object', 'persona_qa': 'persona',
             'repeat': 'repeat', 'sorting_copy': 'sort'}
GRID_ENVS = sorted(ENV_RUNS)
RECALLS = [(1.0, 'rec100'), (0.5, 'rec50'), (0.25, 'rec25'), (0.1, 'rec10')]


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


def _tail_mean(rows, mode, slug):
    sel = rows[-max(1, len(rows) // 10):]
    ks = [k for k in sel[0] if k.startswith(f'{mode}/{slug}/')]
    assert len(ks) == 1, f'{mode}/{slug}: matched {ks}'
    return float(np.mean([r[ks[0]] for r in sel]))


def _agg(paths, mode):
    """(x_mean, y_mean, x_sem, y_sem) over seeds."""
    xs = [_tail_mean(_rows(p), mode, 'hack_freq_hackable') for p in paths]
    ys = [_tail_mean(_rows(p), mode, 'retain') for p in paths]
    sem = lambda a: np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return np.mean(xs), np.mean(ys), sem(xs), sem(ys)


def series_points(env, method):
    """[(recall, x, y, xerr, yerr)] for GR ('graft') or RP ('rp2')."""
    prefix, _dn, rp_sweep, (gr_sweep, gr_pat) = ENV_RUNS[env]
    mode = 'retain_only' if method == 'graft' else 'both'
    pts = []
    for recall, rtag in RECALLS:
        if rtag == 'rec100':
            if method == 'graft':
                paths = _run_dirs(RESP_ROOT, gr_sweep, f'{prefix}_{gr_pat}')
            else:
                paths = _run_dirs(RESP_ROOT, rp_sweep, f'{prefix}_rp2_hf050')
        else:
            try:
                paths = _run_dirs(RECALL_ROOT, '.', f'{prefix}_{method}_{rtag}_')
            except AssertionError:
                if PRELIM:
                    print(f'  [prelim] {prefix}_{method}_{rtag}: SKIPPED (none finished)')
                    continue
                raise
        pts.append((recall,) + _agg(paths, mode))
    return pts


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8))
    for ax, env in zip(axes.flat, GRID_ENVS):
        prefix, dn_sweep, _rp, _gr = ENV_RUNS[env]
        # DN reference (recall-invariant)
        dx, dy, dxe, dye = _agg(_run_dirs(RESP_ROOT, dn_sweep,
                                          f'{prefix}_donothing'), 'both')
        ax.errorbar(dx, dy, xerr=dxe, yerr=dye, fmt='s', ms=8, color=DN_COLOR,
                    ecolor=DN_COLOR, elinewidth=1.1, capsize=3, mew=0.5,
                    mec='white', zorder=6)
        for method, color in (('rp2', RP_COLOR), ('graft', GR_COLOR)):
            pts = series_points(env, method)
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax.plot(xs, ys, color=color, lw=1.3, alpha=0.75, zorder=5)
            for i, (recall, x, y, xe, ye) in enumerate(pts):
                ax.errorbar(x, y, xerr=xe, yerr=ye, fmt='o', color=color,
                            ecolor=color, ms=8 if recall == 1.0 else 5.5,
                            elinewidth=1.0, capsize=2.5, mew=0.5, mec='white',
                            alpha=1.0 if recall == 1.0 else 0.55 + 0.15 * (3 - i),
                            zorder=7)
            # annotate the endpoints
            for recall, x, y, _xe, _ye in (pts[0], pts[-1]):
                ax.annotate(f'{recall:g}', (x, y), textcoords='offset points',
                            xytext=(5, 4), fontsize=7, color=color)
        ax.set_title(ENV_TITLE[env], fontsize=11)
        ax.set_xlim(1.05, -0.05)   # inverted: better (less hacking) to the right
        ax.grid(True, lw=0.4, alpha=0.35)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('hack rate (hackable prompts)', fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('retain reward (all prompts)', fontsize=9)
    handles = [
        plt.Line2D([], [], marker='o', ls='-', color=GR_COLOR, mec='white',
                   mew=0.5, label='GRAFT (deploy), recall 1 → 0.1'),
        plt.Line2D([], [], marker='o', ls='-', color=RP_COLOR, mec='white',
                   mew=0.5, label='Reward penalty 2.0, recall 1 → 0.1'),
        plt.Line2D([], [], marker='s', ls='', color=DN_COLOR, mec='white',
                   mew=0.5, label='No intervention (recall-invariant)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    figs = os.path.join(HERE, 'figs')
    os.makedirs(figs, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figs, f'recall_sweep_pareto6.{ext}'),
                    dpi=160, bbox_inches='tight')
    print('wrote', os.path.join(figs, 'recall_sweep_pareto6.{png,pdf}'))
    # numeric dump
    for env in GRID_ENVS:
        for method in ('rp2', 'graft'):
            for recall, x, y, xe, ye in series_points(env, method):
                print(f'  {env:12s} {method:5s} rec={recall:<5g} '
                      f'hack={x:.3f}±{xe:.3f} retain={y:.3f}±{ye:.3f}')


if __name__ == '__main__':
    main()
