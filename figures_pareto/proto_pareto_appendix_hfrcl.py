"""Appendix figures: hack_frac × rh_detector_recall sweeps. (V2 cosmetics.)

Two figures (one per hack_frac):
  proto_pareto_7envs_gr_rp_appendix_hf050.pdf  (hack_frac = 0.5)
  proto_pareto_7envs_gr_rp_appendix_hf090.pdf  (hack_frac = 0.9)

Each shows 4 GR points and 4 RP points (one per recall ∈ {0.1, 0.25,
0.5, 1.0}), connected with a thin line to suggest the parameter
trajectory.

Note: matrix sweeps used the older env defs, so for sort and persona the
trajectories sit in a slightly different (env-shifted) region than the
canonical anchor; see the caption for details.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import aggregate_hf_rcl
from proto_pareto_style_v2 import (
    SLOT_ENVS, LEGEND_SLOT, ARROW_ENV, STYLES, MARKER_SIZE, BASE_FONT,
    setup_axes, draw_better_arrow, draw_legend, save_figure,
)


RECALLS = [0.1, 0.25, 0.5, 1.0]


def _series_style(key):
    _, color, marker, _ = STYLES[key]
    return color, marker


def _draw_series(ax, env, method_label, hf, key, zorder=6):
    """Plot 4 points across recalls connected by a line, in v2 style."""
    color, marker = _series_style(key)
    xs, ys, xerrs, yerrs = [], [], [], []
    for rcl in RECALLS:
        agg = aggregate_hf_rcl(env, method_label, hf, rcl)
        if agg is None:
            continue
        r_m, r_s, h_m, h_s, _ = agg
        xs.append(h_m); ys.append(r_m)
        xerrs.append(h_s); yerrs.append(r_s)
    if len(xs) >= 2:
        ax.plot(xs, ys, '-', color=color, alpha=0.55,
                linewidth=1.4, zorder=zorder - 1, clip_on=False)
    if xs:
        # clip_on=False so markers at hack=0% (right edge after invert_xaxis)
        # aren't sliced by the panel boundary.
        ax.errorbar(
            xs, ys, xerr=xerrs, yerr=yerrs,
            fmt=marker, color=color, markersize=MARKER_SIZE,
            markerfacecolor=color, markeredgecolor=color,
            markeredgewidth=0.0,
            ecolor=color, elinewidth=1.0, capsize=2, zorder=zorder,
            linestyle='None', clip_on=False,
        )


def _series_legend_handle(key, label):
    color, marker = _series_style(key)
    return Line2D(
        [0], [0], marker=marker, color=color, linestyle='-',
        markerfacecolor=color, markeredgecolor=color,
        markeredgewidth=0.0, markersize=MARKER_SIZE,
        linewidth=1.4, alpha=0.85, label=label,
    )


def make_figure(hf, out_basename):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()

    for slot, env in SLOT_ENVS:
        ax = axes[slot]
        _draw_series(ax, env, 'GR', hf, 'gr', zorder=10)
        _draw_series(ax, env, 'RP', hf, 'rp', zorder=9)
        setup_axes(ax, env, slot)
        if env == ARROW_ENV:
            draw_better_arrow(ax)

    series_handles = [
        _series_legend_handle('gr',
                              'Gradient Routing\n(recall ∈ {0.1, 0.25, 0.5, 1.0})'),
        _series_legend_handle('rp',
                              'Reward Penalty\n(recall ∈ {0.1, 0.25, 0.5, 1.0})'),
    ]
    draw_legend(axes[LEGEND_SLOT], keys=[], extra_handles=series_handles)

    save_figure(fig, out_basename)
    plt.close(fig)


def main():
    make_figure(0.5, 'proto_pareto_7envs_gr_rp_appendix_hf050.pdf')
    make_figure(0.9, 'proto_pareto_7envs_gr_rp_appendix_hf090.pdf')


if __name__ == '__main__':
    main()
