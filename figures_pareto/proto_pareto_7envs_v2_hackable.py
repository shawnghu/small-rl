"""Hackable-only variant of the GR vs RP main figure (proto_pareto_7envs_v2.py).

Identical layout, styling, and series to proto_pareto_7envs_v2.py. The only
difference is the data: retain and hack_freq are restricted to hackable
prompts (the subset where the hack is actually available), rather than
averaged over all eval prompts.

Reads aggregated_cache_hackable.json. Refresh it on the data host with:
    .venv/bin/python figures_pareto/dump_aggregated.py --subset hackable
"""
import json
import os

import matplotlib.pyplot as plt

from proto_pareto_style_v2 import (
    SLOT_ENVS, LEGEND_SLOT, LEGEND_ORDER_V2_MAIN, ARROW_ENV,
    setup_axes, draw_point, draw_better_arrow, draw_legend, save_figure,
)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, 'aggregated_cache_hackable.json')

with open(CACHE_PATH) as _f:
    _CACHE = json.load(_f)


def _agg(env, key):
    """Look up a cached aggregator output as (r_m, r_s, h_m, h_s, n) or None."""
    v = _CACHE[env].get(key)
    return tuple(v) if v else None


def _best_rp(env):
    br = _CACHE[env]['best_rp']
    if br['agg'] is None:
        return None
    return br['label'], tuple(br['agg'])


def draw_env(ax, env):
    draw_point(ax, _agg(env, 'noi'),    'noi',    zorder=8)
    draw_point(ax, _agg(env, 'noi_ro'), 'noi_ro', zorder=8)
    draw_point(ax, _agg(env, 'filt'),   'filt',   zorder=8)
    draw_point(ax, _agg(env, 'verif'),  'verif',  zorder=8)
    br = _best_rp(env)
    if br is not None:
        draw_point(ax, br[1], 'rp_best', zorder=9)
    draw_point(ax, _agg(env, 'gr'),   'gr',   zorder=10)
    draw_point(ax, _agg(env, 'base'), 'base', zorder=8)
    if env == ARROW_ENV:
        draw_better_arrow(ax)


def main():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()

    for slot, env in SLOT_ENVS:
        ax = axes[slot]
        draw_env(ax, env)
        setup_axes(ax, env, slot)

    draw_legend(axes[LEGEND_SLOT], keys=LEGEND_ORDER_V2_MAIN)

    save_figure(fig, 'proto_pareto_7envs_gr_rp_v2_hackable.pdf')


if __name__ == '__main__':
    main()
