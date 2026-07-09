"""V3 of the 7-env per-env pareto panel: V2 + the graft-port 1:16-coherence GR arm.

Same layout/series as proto_pareto_7envs_v2.py (cached baselines: old-canonical
GR, no-int, no-int one-adapter, best Reward Penalty, weak/aggressive filtering,
base model) plus one new series computed LIVE from
output/graft_canon_port_coh32/ (not the cache):

  GR +coh 1:16 (ours, new)  purple — classic routing, lambda=1, balanced renorm +
    split-moment, coherence 1:16 w/ pen-2, NO verified-retain; deployed =
    retain_only channel, aggregated with the same semantics as the cache
    (mean over last 10% of routing_eval rows, subset '').

Note: the cached green 'gr' is the OLD canonical recipe (old routing stack,
verified-retain coherence at the same 1:16 dose); the purple point is the new
stack with the classifier-only recipe.

Run: cd figures_pareto && ../.venv/bin/python proto_pareto_7envs_v3_coh32.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt

from proto_pareto_style_v2 import (
    SLOT_ENVS, LEGEND_SLOT, LEGEND_ORDER_V2_MAIN, ARROW_ENV, STYLES, BASE_FONT,
    _legend_handles_for_keys,
    setup_axes, draw_point, draw_better_arrow, save_figure,
)
import proto_pareto_data as data

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, 'aggregated_cache.json')
COH32_DIR = os.path.join(HERE, '..', 'output', 'graft_canon_port_coh32')

STYLES['gr_coh'] = ('GR +coh 1:16 (ours, new)', '#9467bd', 'o', False)
# Hollow twin = same run, forget adapter NOT ablated (two-adapter config).
# The cached old-canonical GR has no pre-ablation twin: its run dirs live only
# on the original data host (absent locally and on the volume), so only the
# new variant gets the filled/hollow pair.
STYLES['gr_coh_pre'] = ('GR +coh 1:16 (pre-ablation)', '#9467bd', 'o', True)
LEGEND_ORDER_V3 = ['gr_coh', 'gr_coh_pre'] + LEGEND_ORDER_V2_MAIN

with open(CACHE_PATH) as _f:
    _CACHE = json.load(_f)


def _agg(env, key):
    v = _CACHE[env].get(key)
    return tuple(v) if v else None


def _best_rp(env):
    br = _CACHE[env]['best_rp']
    if br['agg'] is None:
        return None
    return br['label'], tuple(br['agg'])


def _coh32_agg(env, mode):
    """(r_m, r_s, h_m, h_s, n) for the coh32 runs of `env` at an adapter mode
    (retain_only = post-ablation, both = pre-ablation), cache-identical
    aggregation via proto_pareto_data."""
    paths = sorted(glob.glob(os.path.join(COH32_DIR, f'{env}*_graft_coh32_pen2_lam1_s*')))
    assert len(paths) == 3, f'{env}: expected 3 coh32 runs, found {len(paths)}'
    return data.aggregate_paths(paths, env, mode)


COH32 = {env: _coh32_agg(env, 'retain_only') for env in data.ENVS}
COH32_PRE = {env: _coh32_agg(env, 'both') for env in data.ENVS}


def draw_env(ax, env):
    draw_point(ax, _agg(env, 'noi'),    'noi',    zorder=8)
    draw_point(ax, _agg(env, 'noi_ro'), 'noi_ro', zorder=8)
    draw_point(ax, _agg(env, 'filt'),   'filt',   zorder=8)
    draw_point(ax, _agg(env, 'verif'),  'verif',  zorder=8)
    br = _best_rp(env)
    if br is not None:
        draw_point(ax, br[1], 'rp_best', zorder=9)
    draw_point(ax, _agg(env, 'gr'),   'gr',   zorder=10)
    draw_point(ax, COH32_PRE[env],    'gr_coh_pre', zorder=10)
    draw_point(ax, COH32[env],        'gr_coh', zorder=11)
    draw_point(ax, _agg(env, 'base'), 'base', zorder=8)
    if env == ARROW_ENV:
        draw_better_arrow(ax)


def main():
    print(f'{"env":<15} {"retain":>7} {"±std":>6} {"hack":>6} {"±std":>6}  (coh32 deployed)')
    for env in data.ENVS:
        r_m, r_s, h_m, h_s, n = COH32[env]
        print(f'{env:<15} {r_m:>7.3f} {r_s:>6.3f} {h_m:>6.3f} {h_s:>6.3f}  n={n}')

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()

    for slot, env in SLOT_ENVS:
        ax = axes[slot]
        draw_env(ax, env)
        setup_axes(ax, env, slot)

    # Compact local legend: 9 entries no longer fit the shared draw_legend's
    # spacing (it overflowed into the persona_qa title).
    lax = axes[LEGEND_SLOT]
    for s in lax.spines.values():
        s.set_visible(False)
    lax.set_xticks([]); lax.set_yticks([])
    lax.legend(handles=_legend_handles_for_keys(LEGEND_ORDER_V3), loc='center',
               frameon=False, fontsize=BASE_FONT * 0.9, handlelength=1.4,
               labelspacing=0.55, borderpad=0.4, bbox_to_anchor=(0.62, 0.5))

    save_figure(fig, 'proto_pareto_7envs_gr_rp_v3_coh32.pdf')
    fig.savefig(os.path.join(HERE, 'figs', 'proto_pareto_7envs_gr_rp_v3_coh32.png'), dpi=150)


if __name__ == '__main__':
    main()
