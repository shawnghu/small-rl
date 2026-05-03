"""Shared layout + universal-baseline plotting for proto_pareto_* figures.

All figures (main + 3 appendix) use:
  - 2x4 grid with envs in slots [0,1,2,4,5,6,7] and a legend slot at index 3
  - Same universal baselines: GR, RP-canonical (or best-RP), base model, verified-only
  - "this way is better" arrow embedded in the legend slot
"""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import (
    ENVS, ENV_TITLE,
    aggregate_anchor, aggregate_base_model, aggregate_verified_only,
    aggregate_no_intervention, aggregate_filter_baseline,
)


# -------- Color / marker conventions --------
GR_COLOR    = '#2ca02c'   # green
RP_COLOR    = '#7d4ba0'   # purple (RP canonical / penalty axis)
V_COLOR     = '#d65f00'   # orange (multiplier axis)
R_COLOR     = '#1f9e89'   # teal (extras-ratio axis)
BASE_COLOR  = '#555555'   # dark grey (base model)
VER_COLOR   = '#c44179'   # magenta (verifiable-only)
NOI_COLOR   = '#1f77b4'   # blue (no-intervention baseline)
FILT_COLOR  = '#bcbd22'   # olive (filter baseline)

GR_MARKER   = 'D'         # diamond
RP_MARKER   = 's'         # square
BASE_MARKER = 'o'         # circle
VER_MARKER  = 'h'         # hexagon
NOI_MARKER  = 'P'         # plus-shape (no intervention)
FILT_MARKER = 'X'         # x-shape (filter baseline)

PRIMARY_SIZE   = 11       # GR / RP-canonical / RP-best
BASELINE_SIZE  = 8        # base / verified-only
SPOKE_SIZE     = 7        # parameter-sweep variants


# -------- Universal baselines (drawn on every figure) --------
def draw_universal_baselines(ax, env, rp_marker_label='Reward Penalty (canonical)',
                              best_rp=None):
    """Draws GR, RP-canonical (or best-RP), base model, verifiable-only,
    no-intervention, filter on ax.

    If best_rp is provided as (label, agg), overrides RP-canonical with it
    and annotates the winner label.
    """
    # Base model (no error bars — single t=0 data point per run)
    bm = aggregate_base_model(env)
    if bm is not None:
        r_m, _, h_m, _, _ = bm
        ax.scatter([h_m], [r_m], marker=BASE_MARKER, s=BASELINE_SIZE**2,
                   c=BASE_COLOR, edgecolors='black', linewidths=0.5,
                   alpha=0.92, zorder=8)

    # No-intervention baseline (with error bars)
    ni = aggregate_no_intervention(env)
    if ni is not None:
        r_m, r_s, h_m, h_s, _ = ni
        ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                    fmt=NOI_MARKER, color=NOI_COLOR, markersize=BASELINE_SIZE,
                    markeredgecolor='black', markeredgewidth=0.5,
                    ecolor=NOI_COLOR, elinewidth=1.0, capsize=2,
                    zorder=8)

    # Filter baseline (with error bars)
    fb = aggregate_filter_baseline(env)
    if fb is not None:
        r_m, r_s, h_m, h_s, _ = fb
        ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                    fmt=FILT_MARKER, color=FILT_COLOR, markersize=BASELINE_SIZE,
                    markeredgecolor='black', markeredgewidth=0.5,
                    ecolor=FILT_COLOR, elinewidth=1.0, capsize=2,
                    zorder=8)

    # Verifiable-only (with error bars)
    vo = aggregate_verified_only(env)
    if vo is not None:
        r_m, r_s, h_m, h_s, _ = vo
        ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                    fmt=VER_MARKER, color=VER_COLOR, markersize=BASELINE_SIZE,
                    markeredgecolor='black', markeredgewidth=0.5,
                    ecolor=VER_COLOR, elinewidth=1.0, capsize=2,
                    zorder=8)

    # RP canonical (or best-RP for main)
    if best_rp is not None:
        label, agg = best_rp
        if agg is not None:
            r_m, r_s, h_m, h_s, _ = agg
            ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                        fmt=RP_MARKER, color=RP_COLOR, markersize=PRIMARY_SIZE,
                        markeredgecolor='black', markeredgewidth=0.6,
                        ecolor=RP_COLOR, elinewidth=1.2, capsize=3,
                        zorder=9)
            # Small annotation indicating which RP variant won
            ax.annotate(label, xy=(h_m, r_m),
                        xytext=(7, -7), textcoords='offset points',
                        fontsize=8, color=RP_COLOR)
    else:
        rp = aggregate_anchor(env, 'RP')
        if rp is not None:
            r_m, r_s, h_m, h_s, _ = rp
            ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                        fmt=RP_MARKER, color=RP_COLOR, markersize=PRIMARY_SIZE,
                        markeredgecolor='black', markeredgewidth=0.6,
                        ecolor=RP_COLOR, elinewidth=1.2, capsize=3,
                        zorder=9)

    # GR canonical
    gr = aggregate_anchor(env, 'GR')
    if gr is not None:
        r_m, r_s, h_m, h_s, _ = gr
        ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                    fmt=GR_MARKER, color=GR_COLOR, markersize=PRIMARY_SIZE,
                    markeredgecolor='black', markeredgewidth=0.6,
                    ecolor=GR_COLOR, elinewidth=1.2, capsize=3,
                    zorder=10)


# -------- Per-axis setup (titles, ticks, labels) --------
def setup_axes(ax, env, slot_idx, n_cols=4):
    """2x4 layout with legend in slot 3.

    - Envs in slots [0,1,2,4,5,6,7]
    - Y-tick labels: leftmost column (slot 0, slot 4)
    - X-tick labels: bottom row (slots 4,5,6,7)
    """
    ax.set_title(ENV_TITLE.get(env, env))
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    if slot_idx % n_cols == 0:
        ax.set_ylabel('Retain Reward')
    else:
        ax.tick_params(labelleft=False)
    if slot_idx >= n_cols:
        ax.set_xlabel('Hack Frequency')
    else:
        ax.tick_params(labelbottom=False)


# -------- Legend slot (slot 3) --------
def draw_legend_slot(ax, extra_handles=None,
                     rp_canonical_label='Reward Penalty (canonical)'):
    """Draw the universal-baseline legend + the better-direction arrow inside ax.

    extra_handles: optional list of additional Line2D legend entries (e.g.,
    parameter-sweep variants for the appendix figures).
    """
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    handles = [
        Line2D([0], [0], marker=GR_MARKER, color='w', markerfacecolor=GR_COLOR,
               markeredgecolor='black', markeredgewidth=0.6, markersize=PRIMARY_SIZE,
               label='Gradient Routing (retain-only)'),
        Line2D([0], [0], marker=RP_MARKER, color='w', markerfacecolor=RP_COLOR,
               markeredgecolor='black', markeredgewidth=0.6, markersize=PRIMARY_SIZE,
               label=rp_canonical_label),
        Line2D([0], [0], marker=FILT_MARKER, color='w', markerfacecolor=FILT_COLOR,
               markeredgecolor='black', markeredgewidth=0.5, markersize=BASELINE_SIZE,
               label='Filter (drop detected, renormalize)'),
        Line2D([0], [0], marker=NOI_MARKER, color='w', markerfacecolor=NOI_COLOR,
               markeredgecolor='black', markeredgewidth=0.5, markersize=BASELINE_SIZE,
               label='No intervention'),
        Line2D([0], [0], marker=VER_MARKER, color='w', markerfacecolor=VER_COLOR,
               markeredgecolor='black', markeredgewidth=0.5, markersize=BASELINE_SIZE,
               label='Train on verifiable-only'),
        Line2D([0], [0], marker=BASE_MARKER, color='w', markerfacecolor=BASE_COLOR,
               markeredgecolor='black', markeredgewidth=0.5, markersize=BASELINE_SIZE,
               label='Base model (untrained)'),
    ]
    if extra_handles:
        handles.extend(extra_handles)

    ax.legend(handles=handles, loc='upper center', frameon=False,
              fontsize=8.5, handlelength=1.5, labelspacing=0.7,
              bbox_to_anchor=(0.5, 1.0))

    # "this way is better" arrow in the lower portion of the legend slot.
    # Use a square delta (delta_x = delta_y) so on a square subplot the
    # arrow appears at 45 degrees; previously the y delta was much smaller
    # than the x delta which produced a near-horizontal arrow.
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate('', xy=(0.80, 0.35), xytext=(0.40, 0.05),
                arrowprops=dict(arrowstyle='-|>', color='black',
                                lw=1.6, mutation_scale=14))
    ax.text(0.6, 0.00, 'this way is better',
            ha='center', va='bottom', fontsize=9, style='italic')


# -------- Output paths --------
_HERE = os.path.dirname(os.path.abspath(__file__))


def save_figure(fig, basename):
    """Save as PDF (preferred for the paper) regardless of the basename's
    extension. Accepts a basename ending in .png or .pdf for backward
    compatibility; output is always .pdf."""
    fig.tight_layout()
    if basename.endswith('.png'):
        basename = basename[:-4] + '.pdf'
    elif not basename.endswith('.pdf'):
        basename = basename + '.pdf'
    local = os.path.join(_HERE, 'figs', basename)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    fig.savefig(local, bbox_inches='tight')
    paper_dest = os.path.expanduser('~/gr-paper/figures/' + basename)
    fig.savefig(paper_dest, bbox_inches='tight')
    print(f'wrote {local} and {paper_dest}')
