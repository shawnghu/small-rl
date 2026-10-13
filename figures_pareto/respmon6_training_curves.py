"""Training curves for the six canonical respmon envs (the paper's
fig:training-curves, figures/training_curves_all_envs.pdf), repointed from
the old prompt-conditioned sweeps to the 2026-07-27 canonical data.

6 envs as rows, 4 metrics as cols (combined reward, retain reward on ALL
prompts, hack freq on detectable prompts, hack freq on blind-spot prompts).
Three series per panel:
  - GRAFT retain_only (green)  — single-adapter deployment eval
  - GRAFT both        (orange) — both adapters active
  - RP both           (purple) — reward penalty at the per-env picked dose

Sweep mapping, run-name conventions, the finished-run guard, and the RP dose
pick (apparent performance = tail-mean combined proxy minus tail-mean monitor
flag rate, complete 3-seed groups only) are all lifted verbatim from
proto_respmon6_rv2canon_sidebyside.py (Jake's box; the authoritative
main-figure script). GR = rv2 canonical recipe (coherence slice from the FULL
batch distribution, coherence_rh_mode=penalty) — NOT vercoh. object_qa =
inverted-monitor variant; sort = the 2026-07-26 framed redesign.

Data: /workspace/respmon-results (mirror rsynced from Jake's box 2026-07-28).
Mean over 3 seeds; band = +/-1 std (ddof 0), matching the paper caption.

Run: cd figures_pareto && ../.venv/bin/python respmon6_training_curves.py
"""
import functools
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

RESP_ROOT = '/workspace/respmon-results'
HERE = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 3

GR_COLOR = '#2ca02c'        # green  (GRAFT retain-only)
GR_BOTH_COLOR = '#d65f00'   # orange (GRAFT both adapters)
RP_COLOR = '#7d4ba0'        # purple (RP both adapters)
DN_COLOR = '#6B7280'        # gray   (do-nothing, both adapters trivially)

# env -> (run-name prefix, (dn_sweep, rp_sweep), (gr_sweep, gr_pat)) —
# the curves-relevant slice of the sidebyside script's ENV_RUNS. The dn_sweep
# entry feeds the do-nothing series (both mode).
_GRCANON = ('respmon_rv2_gr_3seed', 'gr_cls')
ENV_RUNS = {
    'addition_v2': ('addition_v2_syco_respmon_iva',
                    ('respmon_iva_dn_3seed', 'respmon_iva_rp_3seed'), _GRCANON),
    'cities_qa':   ('cities_qa_syco_respmon_iva',
                    ('respmon_iva_dn_3seed', 'respmon_iva_rp_3seed'), _GRCANON),
    'object_qa':   ('object_qa_syco_respmon_iva_inv',
                    ('respmon_ivainv_dn_rp_3seed',) * 2,
                    ('respmon_objinv_gr_canon_3seed', 'gr_cls_canon')),
    'persona_qa':  ('persona_qa_flattery_respmon_bvo',
                    ('respmon_persona_bvo_dn_rp_3seed',) * 2, _GRCANON),
    'repeat':      ('repeat_respmon',
                    ('respmon_repeat_sorting_dn_rp_3seed',) * 2, _GRCANON),
    'sorting_copy': ('sortframed', ('sort_framed',) * 2,
                     ('sort_framed', 'graft_hf')),
}
ENV_TITLE = {'addition_v2': 'addition', 'cities_qa': 'cities',
             'object_qa': 'object', 'persona_qa': 'persona',
             'repeat': 'repeat', 'sorting_copy': 'sort'}
GRID_ENVS = sorted(ENV_RUNS)
_RP_DOSE_SWEEPS = ('respmon_rp_doses_canon_3seed',)

METRICS = [
    ('combined',                'Combined reward'),
    ('retain',                  'Retain reward'),
    ('hack_freq_detectable',    'Hack freq (monitored)'),
    ('hack_freq_undetectable',  'Hack freq (blind-spot)'),
]
SERIES = [
    ('GRAFT (retain-only)',  'gr', 'retain_only', GR_COLOR),
    ('GRAFT (both adapters)', 'gr', 'both',        GR_BOTH_COLOR),
    ('RP (both adapters)',   'rp', 'both',        RP_COLOR),
    ('No intervention',      'dn', 'both',        DN_COLOR),
]


def _run_dirs(sweep, prefix, method_pat):
    root = os.path.join(RESP_ROOT, sweep)
    dirs = sorted(d for d in os.listdir(root)
                  if d.startswith(f'{prefix}_{method_pat}'))
    assert 2 <= len(dirs) <= N_SEEDS, \
        f'{sweep}/{prefix}_{method_pat}*: expected <={N_SEEDS} runs, found {dirs}'
    return [os.path.join(root, d) for d in dirs]


def _rows(path):
    with open(os.path.join(path, 'routing_eval.jsonl')) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert rows, f'empty routing_eval in {path}'
    by_step = {r['step']: r for r in rows}   # keep-last per step (restarts)
    return [by_step[s] for s in sorted(by_step)]


def _key(row, mode, slug):
    ks = [k for k in row if k.startswith(f'{mode}/{slug}/')]
    assert len(ks) == 1, f'{mode}/{slug}: matched {ks}'
    return ks[0]


def _tail_mean(rows, mode, slug):
    sel = rows[-max(1, len(rows) // 10):]
    k = _key(sel[0], mode, slug)
    return float(np.mean([r[k] for r in sel]))


def _apparent_score(paths):
    return float(np.mean([
        _tail_mean(_rows(p), 'both', 'combined')
        - _tail_mean(_rows(p), 'both', 'detected_freq') for p in paths]))


@functools.lru_cache(maxsize=None)
def _finished(path):
    with open(os.path.join(path, 'run_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    return _rows(path)[-1]['step'] >= cfg['max_steps'] - cfg.get('eval_every', 10)


@functools.lru_cache(maxsize=None)
def _best_rp(env):
    prefix, (_, rp_sweep), _gr = ENV_RUNS[env]
    groups, on_disk = defaultdict(list), set()
    for sweep in (rp_sweep,) + _RP_DOSE_SWEEPS:
        root = os.path.join(RESP_ROOT, sweep)
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            m = re.match(re.escape(prefix) + r'_(rp\d+)_', d)
            if not m:
                continue
            on_disk.add(m.group(1))
            p = os.path.join(root, d)
            if os.path.exists(os.path.join(p, 'routing_eval.jsonl')) and _finished(p):
                groups[m.group(1)].append(p)
    assert on_disk, f'no RP runs for {prefix}'
    partial = {t: len(groups.get(t, ())) for t in on_disk
               if len(groups.get(t, ())) != N_SEEDS}
    for tag in partial:
        groups.pop(tag, None)
    if partial:
        print(f'  [rp] {env}: dropped unfinished dose(s) '
              + ', '.join(f'{t} (n={n}/{N_SEEDS})' for t, n in sorted(partial.items())))
    assert groups, f'{env}: no RP dose has {N_SEEDS} finished seeds ({partial})'
    scores = {tag: _apparent_score(ps) for tag, ps in groups.items()}
    best = max(scores, key=scores.get)
    print(f'rp pick {env}: {best}  (apparent: '
          + ', '.join(f'{t}={s:.3f}' for t, s in sorted(scores.items())) + ')')
    return tuple(groups[best])


def _series_paths(env, arm):
    prefix, (dn_sweep, _rp_sweep), (gr_sweep, gr_pat) = ENV_RUNS[env]
    if arm == 'gr':
        return _run_dirs(gr_sweep, prefix, gr_pat)
    if arm == 'dn':
        return _run_dirs(dn_sweep, prefix, 'donothing')
    return list(_best_rp(env))


def _aggregate(paths, mode):
    """metric -> (steps, mean, std) over seeds, prefix-matched keys."""
    out = {}
    per_run = [_rows(p) for p in paths]
    for slug, _t in METRICS:
        by_step = defaultdict(list)
        for rows in per_run:
            try:
                k = _key(rows[0], mode, slug)
            except AssertionError:
                continue
            for r in rows:
                if k in r:
                    by_step[r['step']].append(r[k])
        if not by_step:
            continue
        steps = sorted(by_step)
        out[slug] = (np.array(steps),
                     np.array([np.mean(by_step[s]) for s in steps]),
                     np.array([np.std(by_step[s], ddof=0) for s in steps]))
    return out


def main():
    n = len(GRID_ENVS)
    fig, axes = plt.subplots(n, 4, figsize=(14, 2.4 * n))
    for row, env in enumerate(GRID_ENVS):
        env_series = []
        for label, arm, mode, color in SERIES:
            paths = _series_paths(env, arm)
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
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.015, 1, 1])
    figs = os.path.join(HERE, 'figs')
    os.makedirs(figs, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figs, f'training_curves_all_envs.{ext}'),
                    dpi=160, bbox_inches='tight')
    print('wrote', os.path.join(figs, 'training_curves_all_envs.{png,pdf}'))
    plt.close(fig)


if __name__ == '__main__':
    main()
