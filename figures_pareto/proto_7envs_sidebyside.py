"""7-envs side-by-side composite for the paper's demoted supporting section
(Jake 2026-07-13): ONE pdf, two panels.

Left  = proto_figure1_v3's content (the monitored/unmonitored hack-rate
        cluster scatter, drawn by proto_figure1_v2.draw_scatter — same data,
        same picks, same CI machinery; hosts its own legend).
Right = the 7-env aggregate pareto (proto_pareto_7envs_aggregate.draw_aggregate;
        its own legend — the class sets of the two panels differ).

Run: cd figures_pareto && ../.venv/bin/python proto_7envs_sidebyside.py
"""
import os

import matplotlib.pyplot as plt

# proto_figure1_v2 sets rcParams font 20 at import; draw_aggregate's fonts key
# off proto_pareto_style_v2.BASE_FONT (15) — both panels render legibly at the
# composite size below.
from proto_figure1_v2 import (HERE, ROOT, draw_scatter, legend_handles,
                              print_nocoh_status, print_rp_status)
from proto_pareto_7envs_aggregate import draw_aggregate


def main():
    print_rp_status()
    print_nocoh_status()

    fig = plt.figure(figsize=(17.0, 8.2))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[1.0, 1.0], wspace=0.0)
    TOP, BOT = 0.97, 0.10

    ax_l = sub_l.subplots(1, 1)
    draw_scatter(ax_l)
    ax_l.legend(handles=legend_handles(), loc='lower right', frameon=True)
    sub_l.subplots_adjust(left=0.11, right=0.98, top=TOP, bottom=BOT)

    ax_r = sub_r.subplots(1, 1)
    draw_aggregate(ax_r)
    sub_r.subplots_adjust(left=0.11, right=0.98, top=TOP, bottom=BOT)

    for d in (os.path.join(HERE, 'figs'), os.path.join(ROOT, 'final_figures')):
        os.makedirs(d, exist_ok=True)
        for ext, kw in (('pdf', {}), ('png', {'dpi': 150})):
            out = os.path.join(d, f'proto_7envs_sidebyside.{ext}')
            fig.savefig(out, bbox_inches='tight', pad_inches=0.04, **kw)
            print(f'wrote {out}')


if __name__ == '__main__':
    main()
