"""Monitored vs unmonitored hacking — partial-forget + RoutedAdam variant.

proto_pareto_monitored_partial_forget with the extra class
  GRAFT: partial forget, RoutedAdam
(classic + RoutedAdam shared-v bw2, seeds {1,3,5}, per-seed optimal forget
scale — same retain - 2*hack scoring). Saved under a separate *_radam
filename so the original figure is undisturbed.

Run:
    .venv/bin/python figures_pareto/proto_pareto_monitored_partial_forget_radam.py
"""
import os

import matplotlib.pyplot as plt

from proto_pareto_monitored_partial_forget import draw_scatter, legend_handles

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    draw_scatter(ax, include_radam=True)
    ax.legend(handles=legend_handles(include_radam=True), loc="lower right",
              frameon=True, fontsize=12)
    out_pdf = os.path.join(HERE, "figs", "proto_pareto_monitored_partial_forget_radam.pdf")
    out_png = os.path.join(HERE, "figs", "proto_pareto_monitored_partial_forget_radam.png")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
