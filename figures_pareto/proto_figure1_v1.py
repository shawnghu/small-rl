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

from proto_pareto_monitored_v2 import draw_scatter, legend_handles
from proto_uplift_panel_v1 import draw as draw_uplift

HERE = os.path.dirname(os.path.abspath(__file__))

# One font size for the whole figure; ASCII minus for the panel's negatives.
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False


def main():
    fig = plt.figure(figsize=(17.0, 9.0))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[9.0, 8.0], wspace=0.0)

    # Shared vertical span: both halves' plot areas run top=0.97 .. bottom=0.10.
    # sub_l's axes box is made slightly wider than tall so the aspect='equal'
    # scatter is height-limited and fills that span exactly, aligning its top
    # and bottom with the right-hand panels.
    TOP, BOT = 0.97, 0.10

    # --- left: monitored / unmonitored hack-rate scatter (hosts the legend) ---
    ax_sc = sub_l.subplots(1, 1)
    draw_scatter(ax_sc)
    ax_sc.legend(handles=legend_handles(), loc='lower right', frameon=True)
    sub_l.subplots_adjust(left=0.10, right=0.99, top=TOP, bottom=BOT)

    # --- right: env-averaged training panel (task uplift top, hack rate bottom) ---
    ax_top, ax_bot = sub_r.subplots(2, 1, sharex=True)
    draw_uplift(ax_top, 'retain', subtract_base=True)
    ax_top.set_ylabel('Task performance\nimprovement')
    draw_uplift(ax_bot, 'hack_freq', subtract_base=False)
    ax_bot.set_ylabel('Reward hack rate')
    ax_bot.set_xlabel('Training step')
    sub_r.subplots_adjust(left=0.17, right=0.97, top=TOP, bottom=BOT, hspace=0.07)
    sub_r.align_ylabels([ax_top, ax_bot])

    out = os.path.join(HERE, 'figs', 'proto_figure1_v1.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
