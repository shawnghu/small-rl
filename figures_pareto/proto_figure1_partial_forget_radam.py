"""Composite Figure 1 — partial-forget + RoutedAdam variant.

Same layout as proto_figure1_partial_forget, with one extra class on both
panels: "GRAFT: partial forget, RoutedAdam" (classic routing + RoutedAdam
shared-v, bw2, seeds {1,3,5} — see MODAL_RUNS.md "RoutedAdam-classic").
Saved under separate *_radam filenames so the original figure is undisturbed.

Data: output/gr_forget_scale_eval/canonical_radam_1k_samples/results.jsonl
(left panel + per-seed optima) and canonical_radam_trajectory_optimum/
(right panel trajectories).

Run:
    .venv/bin/python figures_pareto/proto_figure1_partial_forget_radam.py
"""
import os

import matplotlib.pyplot as plt

from proto_pareto_monitored_partial_forget import draw_scatter, legend_handles
from proto_uplift_panel_partial_forget import (
    RADAM_BW1_CLASS, RADAM_BW1_PA_CLASS, RADAM_CLASS, RADAM_PA_CLASS,
    draw as draw_uplift,
)

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["font.size"] = 20
plt.rcParams["axes.unicode_minus"] = False


def main():
    fig = plt.figure(figsize=(17.0, 9.0))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[9.0, 8.0], wspace=0.0)
    TOP, BOT = 0.97, 0.10

    ax_sc = sub_l.subplots(1, 1)
    draw_scatter(ax_sc, include_radam=True)
    ax_sc.legend(handles=legend_handles(include_radam=True), loc="lower right",
                 frameon=True, fontsize=13)
    sub_l.subplots_adjust(left=0.10, right=0.99, top=TOP, bottom=BOT)

    ax_top, ax_bot = sub_r.subplots(2, 1, sharex=True)
    _extras = (RADAM_BW1_CLASS, RADAM_CLASS, RADAM_BW1_PA_CLASS, RADAM_PA_CLASS)
    draw_uplift(ax_top, "retain", subtract_base=True, extra_classes=_extras)
    ax_top.set_ylabel("Task performance\nimprovement")
    draw_uplift(ax_bot, "hack_freq", subtract_base=False, extra_classes=_extras)
    ax_bot.set_ylabel("Reward hack rate")
    ax_bot.set_xlabel("Training step")
    sub_r.subplots_adjust(left=0.17, right=0.97, top=TOP, bottom=BOT, hspace=0.07)
    sub_r.align_ylabels([ax_top, ax_bot])

    out_pdf = os.path.join(HERE, "figs", "proto_figure1_partial_forget_radam.pdf")
    out_png = os.path.join(HERE, "figs", "proto_figure1_partial_forget_radam.png")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
