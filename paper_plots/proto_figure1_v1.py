"""Composite Figure 1.

Left:  monitored-vs-unmonitored hack-rate scatter (proto_pareto_monitored_v2) —
       the "conditional hack vs hack" view, four intervention classes. Hosts
       the shared legend (bottom-right).
Right: env-averaged training panel — task-performance uplift (top) and absolute
       reward-hack rate (bottom), same four classes (proto_uplift_panel_v1).

Both halves share the same four classes, the same colors, and the same font
size (rcParams below); one legend in the scatter serves both.

Run:
    .venv/bin/python figures_pareto/proto_figure1_v1.py
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from proto_pareto_monitored_v2 import draw_scatter, legend_handles
from proto_uplift_panel_v1 import draw as draw_uplift

HERE = os.path.dirname(os.path.abspath(__file__))

# One font size for the whole figure; ASCII minus for the panel's negatives.
plt.rcParams['font.size'] = 22
plt.rcParams['axes.unicode_minus'] = False


def main():
    # Two equal-width subfigures and matched left/right margins so the scatter's
    # square plot region and the right panels' x-width are identical. The left
    # margin must accommodate the right ylabel + tick labels so they don't
    # crowd the scatter to its left — fig_w/margins are sized for that.
    fig = plt.figure(figsize=(19.1, 9.0))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[1.0, 1.0], wspace=0.0)

    TOP, BOT = 0.97, 0.10
    LEFT, RIGHT = 0.17, 0.99  # same on both halves -> identical plot widths

    # --- left: monitored / unmonitored hack-rate scatter (hosts the legend) ---
    ax_sc = sub_l.subplots(1, 1)
    draw_scatter(ax_sc)
    ax_sc.legend(handles=legend_handles(), loc='lower right', frameon=True)
    sub_l.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOT)

    # --- right: env-averaged training panel (task uplift top, hack rate bottom) ---
    ax_top, ax_bot = sub_r.subplots(2, 1, sharex=True)
    draw_uplift(ax_top, 'retain', subtract_base=True)
    ax_top.set_ylabel('Performance uplift')
    draw_uplift(ax_bot, 'hack_freq', subtract_base=False)
    ax_bot.set_ylabel('Reward hack rate')
    ax_bot.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax_bot.set_xlabel('Training step')
    sub_r.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOT, hspace=0.07)
    sub_r.align_ylabels([ax_top, ax_bot])

    out = os.path.join(HERE, 'figs', 'proto_figure1_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
