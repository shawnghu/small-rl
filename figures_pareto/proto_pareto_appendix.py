"""Appendix figures: per-parameter sweeps showing GR + canonical RP +
universal baselines (base model, verified-only) + that parameter's variants.

Three figures saved (each 2x4 with legend in slot 3):
  proto_pareto_7envs_gr_rp_appendix_p.png  (penalty)
  proto_pareto_7envs_gr_rp_appendix_v.png  (multiplier)
  proto_pareto_7envs_gr_rp_appendix_r.png  (extras:main ratio, excluding 0:1
                                             which is now the verified-only baseline)
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import (
    ENVS, aggregate_p, aggregate_v, aggregate_ratio,
)
from proto_pareto_layout import (
    setup_axes, draw_universal_baselines, draw_legend_slot, save_figure,
    SPOKE_SIZE, RP_COLOR,
)


PLOT_SLOTS = [0, 1, 2, 4, 5, 6, 7]
LEGEND_SLOT = 3


def make_figure(spoke_specs, color, out_basename):
    """spoke_specs: list of (label, agg_fn, marker)."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    axes = axes.flatten()

    for env, i in zip(ENVS, PLOT_SLOTS):
        ax = axes[i]
        draw_universal_baselines(ax, env)
        for _label, agg_fn, marker in spoke_specs:
            agg = agg_fn(env)
            if agg is None: continue
            r_m, _, h_m, _, _ = agg
            ax.scatter([h_m], [r_m], marker=marker, s=SPOKE_SIZE**2,
                       c=color, edgecolors='black', linewidths=0.45,
                       alpha=0.92, zorder=6)
        setup_axes(ax, env, i)

    extra_handles = [
        Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
               markeredgecolor='black', markeredgewidth=0.45,
               markersize=SPOKE_SIZE + 1, label=label)
        for label, _agg, marker in spoke_specs
    ]
    draw_legend_slot(axes[LEGEND_SLOT], extra_handles=extra_handles)
    save_figure(fig, out_basename)
    plt.close(fig)


def main():
    # Reward penalty value: anchor=2 + 5, 10
    make_figure(
        spoke_specs=[
            ('reward penalty = 5',  lambda e: aggregate_p(e, 5),  's'),
            ('reward penalty = 10', lambda e: aggregate_p(e, 10), 'D'),
        ],
        color=RP_COLOR,
        out_basename='proto_pareto_7envs_gr_rp_appendix_p.png',
    )

    # Verified-retain sample advantage multiplier: anchor=1 + 2, 5
    make_figure(
        spoke_specs=[
            ('verified-retain multiplier = 2', lambda e: aggregate_v(e, 2), '^'),
            ('verified-retain multiplier = 5', lambda e: aggregate_v(e, 5), '*'),
        ],
        color=RP_COLOR,
        out_basename='proto_pareto_7envs_gr_rp_appendix_v.png',
    )

    # Verifiable:full-distribution-rollout ratio: anchor=1:16 + 1:4, 1:2, 1:1
    # (0:1 = train-on-verifiable-only is its own universal baseline)
    make_figure(
        spoke_specs=[
            ('verifiable : full-distribution rollout = 1:4', lambda e: aggregate_ratio(e, '1:4'), 'p'),
            ('verifiable : full-distribution rollout = 1:2', lambda e: aggregate_ratio(e, '1:2'), 'v'),
            ('verifiable : full-distribution rollout = 1:1', lambda e: aggregate_ratio(e, '1:1'), 'X'),
        ],
        color=RP_COLOR,
        out_basename='proto_pareto_7envs_gr_rp_appendix_r.png',
    )


if __name__ == '__main__':
    main()
