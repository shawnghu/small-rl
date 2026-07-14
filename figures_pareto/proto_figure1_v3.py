"""Figure 1 v3 — v2's monitored/unmonitored scatter ONLY (Jake 2026-07-13:
drop the right-side training panels; otherwise unchanged).

All data and cosmetics are imported from proto_figure1_v2 (same classes, same
classifier-picked deployment, same cluster/CI machinery) so the two versions
can never drift. v2 keeps rendering the composite to figs/ as a working copy;
the camera-ready final_figures/ carries v3.

Run: cd figures_pareto && ../.venv/bin/python proto_figure1_v3.py
"""
import os

import matplotlib.pyplot as plt

from proto_figure1_v2 import (HERE, ROOT, draw_scatter, legend_handles,
                              print_nocoh_status, print_rp_status)


def main():
    print_rp_status()
    print_nocoh_status()
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    draw_scatter(ax)
    ax.legend(handles=legend_handles(), loc='lower right', frameon=True)
    fig.tight_layout()
    for d in (os.path.join(HERE, 'figs'), os.path.join(ROOT, 'final_figures')):
        os.makedirs(d, exist_ok=True)
        for ext, kw in (('pdf', {}), ('png', {'dpi': 150})):
            out = os.path.join(d, f'proto_figure1_v3.{ext}')
            fig.savefig(out, bbox_inches='tight', pad_inches=0.04, **kw)
            print(f'wrote {out}')


if __name__ == '__main__':
    main()
