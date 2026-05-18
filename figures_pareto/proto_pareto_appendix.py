"""Appendix figures: per-parameter sweeps showing GR + canonical RP +
universal baselines + that parameter's variants. (V2 cosmetics.)

Three figures saved (each 2x4 with legend in slot 3):
  proto_pareto_7envs_gr_rp_appendix_p.pdf  (penalty)
  proto_pareto_7envs_gr_rp_appendix_v.pdf  (multiplier)
  proto_pareto_7envs_gr_rp_appendix_r.pdf  (extras:main ratio, excluding 0:1
                                             which is now the verified-only baseline)
"""
import matplotlib.pyplot as plt

from proto_pareto_data import (
    aggregate_anchor, aggregate_base_model, aggregate_no_intervention,
    aggregate_no_intervention_retain_only,
    aggregate_filter_baseline, aggregate_verified_only,
    aggregate_p, aggregate_v, aggregate_ratio,
)
from proto_pareto_style_v2 import (
    SLOT_ENVS, LEGEND_SLOT, LEGEND_ORDER_APPENDIX, ARROW_ENV, STYLES,
    setup_axes, draw_point, draw_better_arrow, draw_legend,
    make_legend_handle, save_figure,
)


# Spoke variants render in the RP color so they read as RP-parameter sweeps.
RP_COLOR = STYLES['rp'][1]


def _draw_universal_appendix(ax, env):
    """7 universal series for appendix figures."""
    draw_point(ax, aggregate_base_model(env),                  'base',   zorder=7)
    draw_point(ax, aggregate_no_intervention(env),             'noi',    zorder=8)
    draw_point(ax, aggregate_no_intervention_retain_only(env), 'noi_ro', zorder=8)
    draw_point(ax, aggregate_filter_baseline(env),             'filt',   zorder=8)
    draw_point(ax, aggregate_verified_only(env),               'verif',  zorder=8)
    draw_point(ax, aggregate_anchor(env, 'RP'),                'rp',     zorder=9)
    draw_point(ax, aggregate_anchor(env, 'GR'),                'gr',     zorder=10)


def make_figure(spoke_specs, color, out_basename):
    """spoke_specs: list of (label, agg_fn, marker)."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()

    for slot, env in SLOT_ENVS:
        ax = axes[slot]
        _draw_universal_appendix(ax, env)
        for _label, agg_fn, marker in spoke_specs:
            draw_point(ax, agg_fn(env), color=color, marker=marker, zorder=7)
        setup_axes(ax, env, slot)
        if env == ARROW_ENV:
            draw_better_arrow(ax)

    extra_handles = [
        make_legend_handle(label, color, marker)
        for label, _agg, marker in spoke_specs
    ]
    draw_legend(axes[LEGEND_SLOT], keys=LEGEND_ORDER_APPENDIX,
                extra_handles=extra_handles)
    save_figure(fig, out_basename)
    plt.close(fig)


def main():
    # Reward penalty value: anchor=2 + 5, 10
    make_figure(
        spoke_specs=[
            ('reward penalty = 5',  lambda e: aggregate_p(e, 5),  '<'),
            ('reward penalty = 10', lambda e: aggregate_p(e, 10), '>'),
        ],
        color=RP_COLOR,
        out_basename='proto_pareto_7envs_gr_rp_appendix_p.pdf',
    )

    # Verified-retain sample advantage multiplier: anchor=1 + 2, 5
    make_figure(
        spoke_specs=[
            ('verified-retain multiplier = 2', lambda e: aggregate_v(e, 2), 'v'),
            ('verified-retain multiplier = 5', lambda e: aggregate_v(e, 5), '*'),
        ],
        color=RP_COLOR,
        out_basename='proto_pareto_7envs_gr_rp_appendix_v.pdf',
    )

    # Verifiable:full-distribution-rollout ratio: anchor=1:16 + 1:4, 1:2, 1:1
    # (0:1 = train-on-verifiable-only is its own universal baseline)
    make_figure(
        spoke_specs=[
            ('verifiable : full-distribution rollout = 1:4', lambda e: aggregate_ratio(e, '1:4'), 'p'),
            ('verifiable : full-distribution rollout = 1:2', lambda e: aggregate_ratio(e, '1:2'), 'h'),
            ('verifiable : full-distribution rollout = 1:1', lambda e: aggregate_ratio(e, '1:1'), 'P'),
        ],
        color=RP_COLOR,
        out_basename='proto_pareto_7envs_gr_rp_appendix_r.pdf',
    )


if __name__ == '__main__':
    main()
