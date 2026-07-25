"""Reviewer-metric variant of proto_7envs_sidebyside (exploratory, 2026-07-24):
right grid reports retain on UNHACKABLE prompts (y) and hack rate on HACKABLE
prompts (x), per reviewer qfQC's disentanglement suggestion. Same runs, same
GR forget-scale pick rule and RP best-penalty selection as v4 (both selection
criteria still use the original overall metrics, so the POINTS move but the
PICKS don't). Left panel unchanged (already hackable-slice).

filt/base are recomputed from the raw runs (routing_eval.jsonl mirrored from
the data host) instead of aggregated_cache.json, replicating the cache's tail
/ first-row protocols with the mixed-subset keys.

Outputs to figs/ only — NOT final_figures (not camera-ready).

Run: cd figures_pareto && ../.venv/bin/python proto_7envs_sidebyside_unhack.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from proto_figure1_v2 import (HERE, ROOT, draw_scatter, legend_handles,
                              print_nocoh_status, print_rp_status)
from proto_pareto_style_v2 import (
    ROW_TOP, ROW_BOT, BASE_FONT,
    _legend_handles_for_keys, draw_point,
)
from proto_pareto_data import ENV_TITLE, DET, ENVS
import proto_pareto_data as data
from proto_pareto_7envs_v4 import (
    series_for_env as series_for_env_orig,
    LEGEND_ORDER_V4, RP_FSDIR, NOCOH_FSDIR, NOINT_FSDIR, PICK_MON_THRESH,
)
import fseval_data as fs

RETAIN_SLUG = 'retain_unhackable'
HACK_SLUG = 'hack_freq_hackable'


# -------- fseval-based series, variant metrics --------
def seed_points_v(fsdir, glob_pat, scale, n_expected=None):
    recs = fs.load_recs(fsdir, glob_pat)
    if n_expected is not None:
        assert len(recs) == n_expected, \
            f"{fsdir}/{glob_pat}: expected {n_expected}, found {len(recs)}"
    return [(fs.pick(r['scales'][scale], RETAIN_SLUG),
             fs.pick(r['scales'][scale], HACK_SLUG)) for r in recs]


def _nocoh_picked_agg_v(env):
    """Same pick rule as v4 (largest scale with monitored hack < thresh),
    variant metrics reported at the picked scale."""
    recs = fs.load_recs(NOCOH_FSDIR, f'{env}*_graft_lam1_s[0-9].json')
    assert len(recs) == 3
    order = [f'{i/10:.1f}' for i in range(11)]
    pts = []
    for rec in recs:
        sm = rec['scales']
        ok = [o for o in order if o in sm
              and fs.pick(sm[o], 'hack_freq_detectable') is not None
              and fs.pick(sm[o], 'hack_freq_detectable') < PICK_MON_THRESH]
        p = max(ok, key=float) if ok else '0.0'
        pts.append((fs.pick(sm[p], RETAIN_SLUG), fs.pick(sm[p], HACK_SLUG)))
    return fs.agg(pts)


def _best_rp_v(env):
    """Penalty chosen exactly as v4 (overall retain - hack at scale 1.0);
    variant metrics reported for that penalty's seeds."""
    recs = fs.load_recs(RP_FSDIR, f'{env}*_rp_noextras_pen*_s*.json')
    if not recs:
        return None
    by_pen = defaultdict(list)
    for rec in recs:
        pen = rec['run_name'].split('_pen')[1].split('_')[0]
        sm = rec['scales']['1.0']
        by_pen[pen].append((fs.pick(sm, 'retain'), fs.pick(sm, 'hack_freq'),
                            fs.pick(sm, RETAIN_SLUG), fs.pick(sm, HACK_SLUG)))
    best_pen = max(by_pen, key=lambda p: (
        sum(r for r, _, _, _ in by_pen[p]) / len(by_pen[p])
        - sum(h for _, h, _, _ in by_pen[p]) / len(by_pen[p])))
    return f'p={best_pen}', fs.agg([(rv, hv) for _, _, rv, hv in by_pen[best_pen]])


# -------- raw-run series (replaces aggregated_cache), variant metrics --------
def _run_tail_v(path, env, mode, tail_frac=True):
    """Mixed-subset mean over the last 10% of routing_eval rows (tail_frac)
    or the first row (base-model protocol). Mirrors data.load_run /
    aggregate_base_model."""
    det = DET[env]
    retain_prefix = f'{mode}/{RETAIN_SLUG}/'
    hack_key = f'{mode}/{HACK_SLUG}/{det}'
    eval_path = os.path.join(ROOT, path, 'routing_eval.jsonl')
    if not os.path.exists(eval_path):
        return None
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    if not rows:
        return None
    sel = rows[-max(1, len(rows) // 10):] if tail_frac else rows[:1]
    rks = [k for k in sel[0] if k.startswith(retain_prefix)]
    if not rks or hack_key not in sel[0]:
        return None
    return (float(np.mean([r[rks[0]] for r in sel])),
            float(np.mean([r[hack_key] for r in sel])))


def _agg_pts(pts):
    pts = [p for p in pts if p is not None]
    return fs.agg(pts)


def filt_v(env):
    if env == 'persona_qa':
        paths = data._persona_pre_paths('filt_3x_renorm_rcl100_hf50')
    else:
        eys = data.EYS_NEW[env]
        paths = [f'output/filter_baseline_7envs/{eys}_filter_baseline_renorm_rcl100_hf50_s{s}'
                 for s in (1, 2, 3)]
    return _agg_pts([_run_tail_v(p, env, 'both') for p in paths])


def base_v(env):
    paths = []
    for cfg in ('RP', 'GR'):
        paths += data.anchor_paths(env, cfg)
    return _agg_pts([_run_tail_v(p, env, 'both', tail_frac=False) for p in paths])


def series_for_env_v(env):
    nocoh_pat = f'{env}*_graft_lam1_s[0-9].json'
    noint_pat = f'{env}*_noint_lam1_s*.json'
    out = [
        ('noi',    fs.agg(seed_points_v(NOINT_FSDIR, noint_pat, '1.0', n_expected=3))),
        ('noi_ro', fs.agg(seed_points_v(NOINT_FSDIR, noint_pat, '0.0', n_expected=3))),
        ('filt',   filt_v(env)),
    ]
    rp = _best_rp_v(env)
    if rp is not None:
        out.append(('rp_best', rp[1]))
    out += [
        ('gr_pre', fs.agg(seed_points_v(NOCOH_FSDIR, nocoh_pat, '1.0', n_expected=3))),
        ('gr',     _nocoh_picked_agg_v(env)),
        ('base',   base_v(env)),
    ]
    return out


# -------- composite (same layout as proto_7envs_sidebyside) --------
GRID_ENVS = sorted(ROW_TOP + ROW_BOT)
N_COLS = 3
ENV_CELLS = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]


def setup_grid_axes(ax, env, row, col):
    ax.set_title(ENV_TITLE.get(env, env), fontsize=19)
    ax.set_box_aspect(1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(labelsize=16)
    if col != 0:
        ax.tick_params(labelleft=False)
    if row != 2:
        ax.tick_params(labelbottom=False)
    elif col != 0:
        ax.set_xticklabels(['0%', '50%', ''])


def main():
    print_rp_status()
    print_nocoh_status()
    have_rp = bool(fs.load_recs(RP_FSDIR, '*.json'))

    # numeric comparison: original vs variant, per env/series
    print(f'{"env":<15} {"series":<8} '
          f'{"r_orig":>7} {"r_new":>7} {"h_orig":>7} {"h_new":>7}')
    for env in GRID_ENVS:
        orig = dict(series_for_env_orig(env))
        var = dict(series_for_env_v(env))
        for key in orig:
            o, v = orig.get(key), var.get(key)
            if o is None or v is None:
                print(f'{env:<15} {key:<8} (missing: orig={o is not None}, new={v is not None})')
                continue
            print(f'{env:<15} {key:<8} {o[0]:>7.3f} {v[0]:>7.3f} {o[2]:>7.3f} {v[2]:>7.3f}')

    fig = plt.figure(figsize=(17.0, 8.2))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[1.0, 0.92], wspace=0.0)
    TOP, BOT = 0.97, 0.10

    ax_l = sub_l.subplots(1, 1)
    draw_scatter(ax_l)
    ax_l.xaxis.label.set_size(25)
    ax_l.yaxis.label.set_size(25)
    ax_l.tick_params(labelsize=20)
    ax_l.legend(handles=legend_handles(), loc='lower right', frameon=True,
                fontsize=20)
    sub_l.subplots_adjust(left=0.11, right=0.98, top=TOP, bottom=BOT)

    gs = sub_r.add_gridspec(3, N_COLS, wspace=0.04, hspace=0.22,
                            left=0.125, right=0.985, top=TOP, bottom=BOT)
    grid_axes = []
    for (row, col), env in zip(ENV_CELLS, GRID_ENVS):
        ax = sub_r.add_subplot(gs[row, col])
        grid_axes.append(ax)
        for z, (key, agg) in enumerate(series_for_env_v(env)):
            if agg is None:
                continue
            draw_point(ax, agg, key, zorder=8 + z)
        setup_grid_axes(ax, env, row, col)
    sub_r.supylabel('Task Performance, unhackable prompts (better →)',
                    fontsize=22, x=0.012, y=(TOP + BOT) / 2)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    lab_bb = inv.transform(ax_l.xaxis.label.get_window_extent())
    y_lab = (lab_bb[0][1] + lab_bb[1][1]) / 2
    left_bb = inv.transform(grid_axes[4].get_window_extent())
    right_bb = inv.transform(grid_axes[6].get_window_extent())
    x_lab = (left_bb[0][0] + right_bb[1][0]) / 2
    fig.text(x_lab, y_lab, 'Hack Frequency, hackable prompts (better →)',
             ha='center', va='center', fontsize=22)

    lax = sub_r.add_subplot(gs[0, 1:])
    for s in lax.spines.values():
        s.set_visible(False)
    lax.set_xticks([]); lax.set_yticks([])
    keys = [k for k in LEGEND_ORDER_V4 if k != 'rp_best' or have_rp]
    lax.legend(handles=_legend_handles_for_keys(keys), loc='center',
               frameon=False, fontsize=16, handlelength=1.4,
               labelspacing=0.55, borderpad=0.1, ncol=1,
               bbox_to_anchor=(0.57, 0.54))

    outdir = os.path.join(HERE, 'figs')
    os.makedirs(outdir, exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 150})):
        out = os.path.join(outdir, f'proto_7envs_sidebyside_unhack.{ext}')
        fig.savefig(out, bbox_inches='tight', pad_inches=0.04, **kw)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
