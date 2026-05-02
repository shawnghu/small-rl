"""Appendix figures: hack_frac × rh_detector_recall sweeps.

Two figures (one per hack_frac):
  proto_pareto_7envs_gr_rp_appendix_hf050.png  (hack_frac = 0.5)
  proto_pareto_7envs_gr_rp_appendix_hf090.png  (hack_frac = 0.9)

Each shows the universal baselines + a series of 4 GR points and 4 RP
points (one per recall ∈ {0.1, 0.25, 0.5, 1.0}), connected with a thin
line to suggest the parameter trajectory.

Note: matrix sweeps used the older env defs (no sort-uniform / persona-3x
upgrades). The canonical anchor / verified-only / base-model baselines on
each panel use the canonical env defs, so for sort and persona the curves
sit in a slightly different (env-shifted) region from the anchor; see the
caption for details.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import (
    ENVS, aggregate_hf_rcl,
)
from proto_pareto_layout import (
    setup_axes, draw_universal_baselines, draw_legend_slot, save_figure,
    SPOKE_SIZE, GR_COLOR, RP_COLOR,
)


PLOT_SLOTS = [0, 1, 2, 4, 5, 6, 7]
LEGEND_SLOT = 3
RECALLS = [0.1, 0.25, 0.5, 1.0]


def make_figure(hf, out_basename, draw_baselines=True):
    """draw_baselines: include the four universal hf=0.5 baselines on every
    panel. For the hf=0.9 figure these are not directly comparable (the
    baselines were trained at hf=0.5), so set this False there."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    axes = axes.flatten()

    for env, i in zip(ENVS, PLOT_SLOTS):
        ax = axes[i]
        if draw_baselines:
            draw_universal_baselines(ax, env)

        # GR series across recalls
        gr_xs, gr_ys = [], []
        for rcl in RECALLS:
            agg = aggregate_hf_rcl(env, 'GR', hf, rcl)
            if agg is None: continue
            r_m, _, h_m, _, _ = agg
            gr_xs.append(h_m); gr_ys.append(r_m)
        if len(gr_xs) >= 2:
            ax.plot(gr_xs, gr_ys, '-', color=GR_COLOR, alpha=0.55,
                    linewidth=1.4, zorder=4)
        for x, y in zip(gr_xs, gr_ys):
            ax.scatter([x], [y], marker='D', s=(SPOKE_SIZE - 1)**2,
                       c=GR_COLOR, edgecolors='black', linewidths=0.4,
                       alpha=0.9, zorder=6)

        # RP series across recalls
        rp_xs, rp_ys = [], []
        for rcl in RECALLS:
            agg = aggregate_hf_rcl(env, 'RP', hf, rcl)
            if agg is None: continue
            r_m, _, h_m, _, _ = agg
            rp_xs.append(h_m); rp_ys.append(r_m)
        if len(rp_xs) >= 2:
            ax.plot(rp_xs, rp_ys, '-', color=RP_COLOR, alpha=0.55,
                    linewidth=1.4, zorder=4)
        for x, y in zip(rp_xs, rp_ys):
            ax.scatter([x], [y], marker='s', s=(SPOKE_SIZE - 1)**2,
                       c=RP_COLOR, edgecolors='black', linewidths=0.4,
                       alpha=0.9, zorder=6)

        setup_axes(ax, env, i)

    extra_handles = [
        Line2D([0], [0], marker='D', color=GR_COLOR, linestyle='-',
               markeredgecolor='black', markeredgewidth=0.4, markersize=SPOKE_SIZE - 1,
               linewidth=1.4, alpha=0.7,
               label='Gradient Routing (recall ∈ {0.1, 0.25, 0.5, 1.0})'),
        Line2D([0], [0], marker='s', color=RP_COLOR, linestyle='-',
               markeredgecolor='black', markeredgewidth=0.4, markersize=SPOKE_SIZE - 1,
               linewidth=1.4, alpha=0.7,
               label='Reward Penalty (recall ∈ {0.1, 0.25, 0.5, 1.0})'),
    ]
    if draw_baselines:
        draw_legend_slot(axes[LEGEND_SLOT], extra_handles=extra_handles)
    else:
        # Minimal legend: just the two series, with the better-direction arrow.
        from proto_pareto_layout import draw_legend_slot as _full_legend
        ax = axes[LEGEND_SLOT]
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(handles=extra_handles, loc='upper center', frameon=False,
                  fontsize=10, handlelength=2.0, labelspacing=0.9,
                  bbox_to_anchor=(0.5, 0.95))
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.annotate('', xy=(0.80, 0.35), xytext=(0.40, 0.05),
                    arrowprops=dict(arrowstyle='-|>', color='black',
                                    lw=1.6, mutation_scale=14))
        ax.text(0.6, 0.00, 'this way is better',
                ha='center', va='bottom', fontsize=9, style='italic')
    save_figure(fig, out_basename)
    plt.close(fig)


def main():
    make_figure(0.5, 'proto_pareto_7envs_gr_rp_appendix_hf050.png',
                draw_baselines=False)
    make_figure(0.9, 'proto_pareto_7envs_gr_rp_appendix_hf090.png',
                draw_baselines=False)


if __name__ == '__main__':
    main()
