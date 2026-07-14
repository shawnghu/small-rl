"""Shared v2 styling for proto_pareto figures (main + appendix).

Centralizes the v2 cosmetic conventions used in proto_pareto_7envs_v2.py:
  - Uniform marker size (no big-vs-small split for primary vs baselines)
  - Marker edges colored to match the marker (no black outlines)
  - Hollow markers (e.g. base model) get an actual edge stroke
  - Fonts +50% (BASE_FONT = 15pt via rcParams)
  - 2x4 grid with envs in slots [0,1,2,4,5,6,7] and legend in slot 3
  - 'better' green arrow drawn inside the repeat_extra subplot (top-left
    of the data area), not in the legend slot
  - Y-axis = "Target Task Performance"; X-axis = "Unintended Behavior
    Frequency" with PercentFormatter
  - .pdf output to local figs/ + ~/gr-paper/figures/ if present
"""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from proto_pareto_data import ENV_TITLE


# -------- Fonts --------
BASE_FONT = 15
plt.rcParams['font.size'] = BASE_FONT


# -------- Layout --------
ROW_TOP = ['repeat_extra', 'sorting_copy', 'topic_contains']
ROW_BOT = ['addition_v2', 'cities_qa', 'object_qa', 'persona_qa']
SLOT_ENVS = list(zip([0, 1, 2, 4, 5, 6, 7], ROW_TOP + ROW_BOT))
LEGEND_SLOT = 3
ARROW_ENV = 'repeat_extra'   # which env's panel hosts the 'better' arrow


# -------- Style table: key -> (label, hex, marker, hollow) --------
# Unified color scheme (Jake 2026-07-09, ALL final figures): reward penalty =
# red, GR deployed = green, GR both-adapters = blue, forget-adapter-only =
# dark red dotted; everything else unchanged.
# Label renames (Jake 2026-07-13): noi_ro describes what the series IS — a
# no-routing run with the (arbitrary) forget half of the adapter neurons
# ablated; rp_best drops "(best)"; filt drops "Weak".
STYLES = {
    'gr':       ('Gradient Routing (ours)',         '#2ca02c', 'o', False),
    'noi':      ('No intervention',                  '#9690a8', 'X', False),
    'noi_ro':   ('Randomly ablate 50% of adapter neurons', '#9690a8', 'X', True),
    'rp':       ('Reward Penalty',                   '#d62728', 's', False),
    'rp_best':  ('Reward Penalty',                   '#d62728', 's', False),
    'filt':     ('Filtering',                        '#b09680', 'D', False),
    'verif':    ('Aggressive Filtering',             '#b08490', '^', False),
    'base':     ('Base model',                       '#444444', 'o', True),
}

LEGEND_ORDER_V2_MAIN = ['gr', 'noi', 'noi_ro', 'rp', 'filt', 'verif', 'base']
LEGEND_ORDER_APPENDIX = ['gr', 'rp', 'filt', 'noi', 'noi_ro', 'verif', 'base']

MARKER_SIZE = 17           # uniform across all icons
HOLLOW_EDGE_LW = 2.0       # only used for hollow markers
ARROW_COLOR = 'black'  # (Jake 2026-07-13; was GR-green)


# -------- Marker helpers --------
def _resolve_style(key=None, *, color=None, marker=None, hollow=False):
    if key is not None:
        _, color, marker, hollow = STYLES[key]
    face = 'white' if hollow else color
    edge_color = color
    edge_w = HOLLOW_EDGE_LW if hollow else 0.0
    return color, marker, face, edge_color, edge_w


def draw_point(ax, agg, key=None, *, color=None, marker=None, hollow=False,
               zorder=10, capsize=3):
    """Draw a single Pareto point (mean ± std error bars) using v2 styling.
    Use `key` for a registered series; pass `color`/`marker` directly for
    appendix spoke variants."""
    if agg is None:
        return
    r_m, r_s, h_m, h_s, _ = agg
    color, marker, face, edge_color, edge_w = _resolve_style(
        key=key, color=color, marker=marker, hollow=hollow,
    )
    ax.errorbar(
        [h_m], [r_m], xerr=[h_s], yerr=[r_s],
        fmt=marker, color=color, markersize=MARKER_SIZE,
        markerfacecolor=face, markeredgecolor=edge_color,
        markeredgewidth=edge_w,
        ecolor=color, elinewidth=1.2, capsize=capsize, zorder=zorder,
    )


# -------- Per-axis setup --------
def setup_axes(ax, env, slot_idx, n_cols=4):
    ax.set_title(ENV_TITLE.get(env, env))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    # Lock to 5 evenly spaced ticks so layouts of different widths render
    # consistently (matplotlib's auto-locator otherwise picks 5 or 6
    # depending on per-panel width).
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if slot_idx % n_cols == 0:
        ax.set_ylabel('Target Task Performance')
    else:
        ax.tick_params(labelleft=False)
    if slot_idx >= n_cols:
        ax.set_xlabel('Unintended Behavior Frequency')
    else:
        ax.tick_params(labelbottom=False)


# -------- 'Better' arrow inside the repeat_extra subplot --------
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
    LABEL = 0.68
    ax.text(LABEL, LABEL, 'better', transform=ax.transAxes,
            fontsize=BASE_FONT * 0.95, style='italic', color=ARROW_COLOR,
            ha='center', va='center')


# -------- Legend slot --------
def make_legend_handle(label, color, marker, hollow=False):
    face = 'white' if hollow else color
    edge_w = HOLLOW_EDGE_LW if hollow else 0.0
    return Line2D(
        [0], [0], marker=marker, color='w',
        markerfacecolor=face, markeredgecolor=color,
        markeredgewidth=edge_w, markersize=MARKER_SIZE,
        label=label,
    )


def _legend_handles_for_keys(keys):
    return [
        make_legend_handle(STYLES[k][0], STYLES[k][1], STYLES[k][2], hollow=STYLES[k][3])
        for k in keys
    ]


def draw_legend(ax, *, keys=None, extra_handles=None):
    """Draw the legend slot (slot 3). The 'better' arrow lives in the
    repeat_extra panel; this slot is purely the series legend."""
    if keys is None:
        keys = LEGEND_ORDER_V2_MAIN
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    handles = _legend_handles_for_keys(keys)
    if extra_handles:
        handles.extend(extra_handles)
    # Push the legend bbox slightly right of slot center so the marker
    # column visually lands inside the rightmost panel column rather than
    # bleeding leftward toward the previous column. (loc='center right'
    # ended up shifting the LEFT edge of an oversized bbox FURTHER left,
    # which was the wrong direction.)
    ax.legend(handles=handles, loc='center', frameon=False,
              fontsize=BASE_FONT * 1.05, handlelength=1.6,
              labelspacing=1.0, borderpad=1.0,
              bbox_to_anchor=(0.65, 0.5))


# -------- Output --------
_HERE = os.path.dirname(os.path.abspath(__file__))


def save_figure(fig, basename):
    """Save as PDF to local figs/ and (if present) ~/gr-paper/figures/.
    Accepts a basename ending in .png or .pdf for backward compatibility;
    output is always .pdf."""
    # Drive inter-subplot gutters down further. tight_layout's pad/w_pad/
    # h_pad are in font-size units (BASE_FONT=15pt). We then override with
    # absolute wspace/hspace to remove residual gutter that tight_layout
    # reserves for tick-label clearance. wspace must leave room for the
    # rightmost tick label of one panel and the leftmost of the next.
    fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3)
    # wspace gives a hair more room so rightmost x-tick labels don't get
    # clipped where one panel ends and the next begins. hspace must stay
    # large enough that the bottom-row titles don't collide with the
    # top-row panel borders.
    fig.subplots_adjust(wspace=0.18, hspace=0.16)
    if basename.endswith('.png'):
        basename = basename[:-4] + '.pdf'
    elif not basename.endswith('.pdf'):
        basename = basename + '.pdf'
    local = os.path.join(_HERE, 'figs', basename)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    # pad_inches small enough to halve the bottom whitespace; large enough
    # that the rightmost x-tick label / x-axis axis label aren't clipped.
    fig.savefig(local, bbox_inches='tight', pad_inches=0.03)
    paper_dest = os.path.expanduser('~/gr-paper/figures/' + basename)
    if os.path.isdir(os.path.dirname(paper_dest)):
        fig.savefig(paper_dest, bbox_inches='tight', pad_inches=0.03)
        print(f'wrote {local} and {paper_dest}')
    else:
        print(f'wrote {local}')
