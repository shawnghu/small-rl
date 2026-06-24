"""V2 of the GR vs RP main figure.

Differences from proto_pareto_7envs.py:
  - New env layout: top row [repeat_extra, sort, topic], bottom row
    [addition, cities_qa, object_qa, persona_qa]; legend in slot 3 (top-right).
  - Restyled markers per request: same size for all icons, no black outlines.
  - Series set: GR, No intervention, No intervention (one adapter), Reward
    Penalty, Weak Filtering, Aggressive Filtering, Base model.
  - Fonts +50% via rcParams.
  - 'better' arrow drawn in the repeat_extra panel along the diagonal,
    not in the legend slot.

Style + layout primitives live in proto_pareto_style_v2.py and are
shared with the appendix figures. Data is loaded from a JSON cache so
this script can render locally without access to output/. To refresh
the cache, run figures_pareto/dump_aggregated.py on the data host.
"""
import json
import os

import matplotlib.pyplot as plt

from proto_pareto_style_v2 import (
    SLOT_ENVS, LEGEND_SLOT, LEGEND_ORDER_V2_MAIN, ARROW_ENV,
    setup_axes, draw_point, draw_better_arrow, draw_legend, save_figure,
)


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, 'aggregated_cache.json')

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
    draw_point(ax, _agg(env, 'gr'),    'gr',    zorder=10)
    draw_point(ax, _agg(env, 'gr_pf'), 'gr_pf', zorder=11)
    draw_point(ax, _agg(env, 'gr_pf_excl'), 'gr_pf_excl', zorder=12)
    draw_point(ax, _agg(env, 'base'),  'base',  zorder=8)
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

    save_figure(fig, 'proto_pareto_7envs_gr_rp_v2.pdf')
    fig.savefig(os.path.join(HERE, 'figs', 'proto_pareto_7envs_gr_rp_v2.png'),
                dpi=140, bbox_inches='tight', pad_inches=0.03)


if __name__ == '__main__':
    main()
