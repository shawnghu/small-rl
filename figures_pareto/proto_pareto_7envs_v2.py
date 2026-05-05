"""V2 of the GR vs RP main figure.

Differences from proto_pareto_7envs.py:
  - New env layout: top row [repeat_extra, sort, topic], bottom row
    [addition, cities_qa, object_qa, persona_qa]; legend in slot 3 (top-right).
  - Restyled markers per request: same size for all icons, no black outlines.
  - New series set: GR, No intervention, Reward Penalty (= GA), Classifier
    Filtering, Qwen3-32B (placeholder — wire data in aggregate_qwen3_32b).
  - Fonts +50% via rcParams.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

import json
import os

from proto_pareto_data import ENV_TITLE


# Path to the JSON cache produced by figures_pareto/dump_aggregated.py on the
# data host. Pre-aggregated so this script can render locally without output/.
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


# Fonts +50%. matplotlib default is 10pt → 15pt across all text.
BASE_FONT = 15
plt.rcParams['font.size'] = BASE_FONT


# Layout: 2x4 grid; slot 3 (top-right) is the legend.
ROW_TOP = ['repeat_extra', 'sorting_copy', 'topic_contains']
ROW_BOT = ['addition_v2', 'cities_qa', 'object_qa', 'persona_qa']
SLOT_ENVS = list(zip([0, 1, 2, 4, 5, 6, 7], ROW_TOP + ROW_BOT))
LEGEND_SLOT = 3


# Style table: key -> (label, hex, marker, hollow)
STYLES = {
    'gr':    ('Gradient Routing (ours)', '#2ca02c', 'o', False),
    'noi':   ('No intervention',     '#9690a8', 'X', False),
    'rp':    ('Reward Penalty',      '#8090a0', 's', False),
    'filt':  ('Weak Filtering',      '#b09680', 'D', False),
    'verif': ('Aggressive Filtering', '#b08490', '^', False),
    'base':  ('Base model',          '#444444', 'o', True),
}
LEGEND_ORDER = ['gr', 'noi', 'rp', 'filt', 'verif', 'base']

MARKER_SIZE = 17          # uniform across all icons
HOLLOW_EDGE_LW = 2.0      # only used for hollow markers (base-model circle)
ARROW_COLOR = '#2ca02c'   # green for "this way is better" arrow


def _marker_style(key):
    label, color, marker, hollow = STYLES[key]
    face = 'white' if hollow else color
    edge_color = color           # never black
    edge_w = HOLLOW_EDGE_LW if hollow else 0.0
    return label, color, marker, face, edge_color, edge_w


def _draw_point(ax, agg, key, zorder=10):
    if agg is None:
        return
    r_m, r_s, h_m, h_s, _ = agg
    _, color, marker, face, edge_color, edge_w = _marker_style(key)
    ax.errorbar(
        [h_m], [r_m], xerr=[h_s], yerr=[r_s],
        fmt=marker, color=color, markersize=MARKER_SIZE,
        markerfacecolor=face, markeredgecolor=edge_color,
        markeredgewidth=edge_w,
        ecolor=color, elinewidth=1.2, capsize=3, zorder=zorder,
    )


def setup_axes(ax, env, slot_idx, n_cols=4):
    ax.set_title(ENV_TITLE.get(env, env))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if slot_idx % n_cols == 0:
        ax.set_ylabel('Target Task Performance')
    else:
        ax.tick_params(labelleft=False)
    if slot_idx >= n_cols:
        ax.set_xlabel('Unintended Behavior Frequency')
    else:
        ax.tick_params(labelbottom=False)


def draw_better_arrow(ax):
    """Green 'better' arrow along the panel's bottom-left → top-right diagonal.
    Endpoints are at x_frac == y_frac so the arrow lies exactly on the square
    diagonal. (X-axis is inverted, so display top-right = low hack-freq + high
    retain = best.)"""
    HEAD = 0.62
    TAIL = 0.12
    ax.annotate(
        '', xy=(HEAD, HEAD), xytext=(TAIL, TAIL),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='-|>', color=ARROW_COLOR, lw=2.5,
                        mutation_scale=22, capstyle='round'),
    )
    # Place the word 'better' so its center sits on the line extending past
    # the arrow head along the diagonal (x_frac == y_frac).
    LABEL = 0.68
    ax.text(LABEL, LABEL, 'better', transform=ax.transAxes,
            fontsize=BASE_FONT * 0.95, style='italic', color=ARROW_COLOR,
            ha='center', va='center')


def draw_env(ax, env):
    _draw_point(ax, _agg(env, 'noi'), 'noi', zorder=8)
    _draw_point(ax, _agg(env, 'filt'), 'filt', zorder=8)
    _draw_point(ax, _agg(env, 'verif'), 'verif', zorder=8)
    br = _best_rp(env)
    if br is not None:
        _draw_point(ax, br[1], 'rp', zorder=9)
    _draw_point(ax, _agg(env, 'gr'), 'gr', zorder=10)
    _draw_point(ax, _agg(env, 'base'), 'base', zorder=8)
    if env == 'repeat_extra':
        draw_better_arrow(ax)


def draw_legend(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    handles = []
    for key in LEGEND_ORDER:
        label, color, marker, face, edge_color, edge_w = _marker_style(key)
        handles.append(Line2D(
            [0], [0], marker=marker, color='w',
            markerfacecolor=face, markeredgecolor=edge_color,
            markeredgewidth=edge_w, markersize=MARKER_SIZE,
            label=label,
        ))
    ax.legend(handles=handles, loc='center', frameon=False,
              fontsize=BASE_FONT * 1.05, handlelength=1.6,
              labelspacing=1.0, borderpad=1.0,
              bbox_to_anchor=(0.5, 0.5))


def main():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()

    for slot, env in SLOT_ENVS:
        ax = axes[slot]
        draw_env(ax, env)
        setup_axes(ax, env, slot)

    draw_legend(axes[LEGEND_SLOT])

    fig.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, 'figs', 'proto_pareto_7envs_gr_rp_v2.pdf')
    fig.savefig(out, bbox_inches='tight')
    paper_dest = os.path.expanduser('~/gr-paper/figures/proto_pareto_7envs_gr_rp_v2.pdf')
    if os.path.isdir(os.path.dirname(paper_dest)):
        fig.savefig(paper_dest, bbox_inches='tight')
        print(f'wrote {out} and {paper_dest}')
    else:
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
