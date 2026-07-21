"""7-envs side-by-side composite for the paper's demoted supporting section
(Jake 2026-07-13): ONE pdf, two panels. Right side reworked 2026-07-21 (Jake):
per-env panels instead of the aggregate/mean pareto.

Left  = proto_figure1_v3's content (the monitored/unmonitored hack-rate
        cluster scatter, drawn by proto_figure1_v2.draw_scatter — same data,
        same picks, same CI machinery; hosts its own legend).
Right = 3x3 grid: the 7 per-env pareto panels (proto_pareto_7envs_v4's
        series_for_env — same series, same picks), envs in row-major slots
        0-6; the legend occupies the last two cells (gs[2, 1:]).

Run: cd figures_pareto && ../.venv/bin/python proto_7envs_sidebyside.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# proto_figure1_v2 sets rcParams font 20 at import; the right grid's fonts key
# off proto_pareto_style_v2.BASE_FONT (15) — both panels render legibly at the
# composite size below.
from proto_figure1_v2 import (HERE, ROOT, draw_scatter, legend_handles,
                              print_nocoh_status, print_rp_status)
from proto_pareto_style_v2 import (
    ROW_TOP, ROW_BOT, BASE_FONT,
    _legend_handles_for_keys, draw_point,
)
from proto_pareto_data import ENV_TITLE
from proto_pareto_7envs_v4 import series_for_env, LEGEND_ORDER_V4, RP_FSDIR
import fseval_data as fs

# Alphabetical env order; the legend occupies the top-right two cells, so
# envs fill cells (0,0) then rows 1-2 in row-major order.
GRID_ENVS = sorted(ROW_TOP + ROW_BOT)   # 7 envs, alphabetical
N_COLS = 3
ENV_CELLS = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]


def setup_grid_axes(ax, env, row, col):
    """Per-panel cosmetics for the 3x3 grid. Same conventions as
    proto_pareto_style_v2.setup_axes but with label placement for the
    3-column layout (legend occupies the top-right two cells; the bottom
    row is full, so it carries all x labels) and sparser x-ticks for the
    narrower panels."""
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
        # Non-leftmost columns: drop the 100% edge label so adjacent panels'
        # 0%/100% labels don't collide in the tight gutters.
        ax.set_xticklabels(['0%', '50%', ''])


def main():
    print_rp_status()
    print_nocoh_status()
    have_rp = bool(fs.load_recs(RP_FSDIR, '*.json'))
    if not have_rp:
        print('WARNING: no-extras RP fseval not found -> right grid renders '
              'WITHOUT an RP series')

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
        for z, (key, agg) in enumerate(series_for_env(env)):
            draw_point(ax, agg, key, zorder=8 + z)
        setup_grid_axes(ax, env, row, col)
    # Shared axis labels for the grid (per-panel labels would repeat 3x).
    sub_r.supylabel('Target Task Performance (better →)', fontsize=25,
                    x=0.012, y=(TOP + BOT) / 2)
    # Placed at EXACTLY the left panel's x-label height (measured from the
    # rendered canvas), horizontally centered under column 0 (persona_qa) so
    # it reads as the plots' label, not the legend's.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    lab_bb = inv.transform(ax_l.xaxis.label.get_window_extent())
    y_lab = (lab_bb[0][1] + lab_bb[1][1]) / 2
    # Bottom row is full again -> center the label across all three columns.
    left_bb = inv.transform(grid_axes[4].get_window_extent())
    right_bb = inv.transform(grid_axes[6].get_window_extent())
    x_lab = (left_bb[0][0] + right_bb[1][0]) / 2
    fig.text(x_lab, y_lab, 'Unintended Behavior Frequency (better →)',
             ha='center', va='center', fontsize=25)

    lax = sub_r.add_subplot(gs[0, 1:])
    for s in lax.spines.values():
        s.set_visible(False)
    lax.set_xticks([]); lax.set_yticks([])
    keys = [k for k in LEGEND_ORDER_V4 if k != 'rp_best' or have_rp]
    lax.legend(handles=_legend_handles_for_keys(keys), loc='center',
               frameon=False, fontsize=16, handlelength=1.4,
               labelspacing=0.55, borderpad=0.1, ncol=1,
               bbox_to_anchor=(0.57, 0.54))

    for d in (os.path.join(HERE, 'figs'), os.path.join(ROOT, 'final_figures')):
        os.makedirs(d, exist_ok=True)
        for ext, kw in (('pdf', {}), ('png', {'dpi': 150})):
            out = os.path.join(d, f'proto_7envs_sidebyside.{ext}')
            fig.savefig(out, bbox_inches='tight', pad_inches=0.04, **kw)
            print(f'wrote {out}')


if __name__ == '__main__':
    main()
