"""Aggregate pareto scatter: mean over the 7 canonical envs, one panel.

Companion to proto_pareto_7envs_v4.py — SAME series, SAME styling (style_v2
markers/colors, inverted percent x-axis), collapsed to a single axes so the
cross-env trend reads directly:

  faint markers = per-env means (over seeds); solid marker = mean over the 7
  envs with SEM half-width per axis (over the 7 env means).

Same retain-classifier-free series set as v4: GR coh32 deployed + pre-ablation
(hollow), new-stack no-int + one-adapter, no-extras RP best (when the sweep
exists), weak filtering (cached), base model (cached).

Run: cd figures_pareto && ../.venv/bin/python proto_pareto_7envs_aggregate.py
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

from proto_pareto_style_v2 import (
    STYLES, BASE_FONT, MARKER_SIZE, HOLLOW_EDGE_LW,
    _legend_handles_for_keys,
)
import proto_pareto_data as data
import fseval_data as fs
from proto_pareto_7envs_v4 import (
    _CACHE, _best_rp, COH32_FSDIR, NOINT_FSDIR, RP_FSDIR, LEGEND_ORDER_V4,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# GR arm switched from coh32 (fixed retain-only ablation) to NO-COHERENCE with
# per-seed classifier-picked forget scale (Jake 2026-07-09), matching the
# countdown deployment methodology. Filled 'gr' = pick; hollow 'gr_pre' = both.
NOCOH_FSDIR = 'output/graft_canon_port_fseval'
PICK_MON_THRESH = 0.02  # highest forget scale with monitored hack < this


def _nocoh_picked_seedpoints(env):
    """Per-seed (retain, hack) at the classifier-picked forget scale: highest
    scale whose monitored (detectable) hack < PICK_MON_THRESH, else fs0."""
    recs = fs.load_recs(NOCOH_FSDIR, f'{env}*_graft_lam1_s[0-9].json')
    assert len(recs) == 3, f"{env}: expected 3 nocoh fseval files, found {len(recs)}"
    order = [f'{i/10:.1f}' for i in range(11)]
    out = []
    for rec in recs:
        sm = rec['scales']
        ok = [o for o in order if o in sm
              and fs.pick(sm[o], 'hack_freq_detectable') is not None
              and fs.pick(sm[o], 'hack_freq_detectable') < PICK_MON_THRESH]
        p = max(ok, key=float) if ok else '0.0'
        out.append((fs.pick(sm[p], 'retain'), fs.pick(sm[p], 'hack_freq')))
    return out


def env_means(key):
    """Per-env (hack_mean, retain_mean) points for one series key."""
    pts = []
    for env in data.ENVS:
        if key == 'gr':
            a = fs.agg(_nocoh_picked_seedpoints(env))
        elif key == 'gr_pre':
            a = fs.agg(fs.seed_points(NOCOH_FSDIR, f'{env}*_graft_lam1_s[0-9].json',
                                      '1.0', n_expected=3))
        elif key in ('noi', 'noi_ro'):
            scale = '1.0' if key == 'noi' else '0.0'
            a = fs.agg(fs.seed_points(NOINT_FSDIR, f'{env}*_noint_lam1_s*.json',
                                      scale, n_expected=3))
        elif key == 'rp_best':
            rp = _best_rp(env)
            a = rp[1] if rp else None
        else:  # cached series: filt, base
            v = _CACHE[env].get(key)
            a = tuple(v) if v else None
        if a is not None:
            pts.append((a[2], a[0]))   # (hack, retain)
    return np.array(pts)


def draw_aggregate(ax):
    """Draw the 7-env aggregate pareto panel onto `ax` (series, limits, labels,
    legend — everything except the figure-level note/save). Factored out so
    proto_7envs_sidebyside.py can host it next to the monitored/unmonitored
    scatter."""
    have_rp = bool(fs.load_recs(RP_FSDIR, '*.json'))
    keys = [k for k in LEGEND_ORDER_V4 if k != 'rp_best' or have_rp]

    print(f'{"series":<10} {"hack":>14} {"retain":>16}  n_envs')
    for key in keys:
        label, color, marker, hollow = STYLES[key]
        pts = env_means(key)
        xs, ys = pts[:, 0], pts[:, 1]
        n = len(xs)
        # SEM over env means (switched from 95% t-CI per Jake 2026-07-09; the
        # hf50 countdown scatter is SEM too, so the conventions now match).
        x_ci = float(np.std(xs, ddof=1) / np.sqrt(n))
        y_ci = float(np.std(ys, ddof=1) / np.sqrt(n))
        print(f'{key:<10} {xs.mean():.3f} +/- {x_ci:.3f}  {ys.mean():.3f} +/- {y_ci:.3f}  {n}')
        ax.scatter(xs, ys, s=(MARKER_SIZE * 0.55) ** 2,
                   facecolors='none' if hollow else color,
                   edgecolors=color, linewidths=1.3 if hollow else 0.0,
                   alpha=0.35, zorder=3, clip_on=False)
        ax.errorbar(xs.mean(), ys.mean(), xerr=x_ci, yerr=y_ci, fmt=marker,
                    markersize=MARKER_SIZE, color=color,
                    markerfacecolor='white' if hollow else color,
                    markeredgecolor=color,
                    markeredgewidth=HOLLOW_EDGE_LW if hollow else 0.0,
                    ecolor=color, elinewidth=1.6, capsize=5, capthick=1.6,
                    zorder=6)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    # The panel's diagonal 'better' arrow needs an empty corner; this axes is
    # dense, so the better-direction hint lives in the axis labels instead.
    ax.set_xlabel('Unintended Behavior Frequency (better →)')
    # Vertical label: glyphs rotate with the text, so "→" points UP on screen.
    ax.set_ylabel('Target Task Performance (better →)')
    ax.legend(handles=_legend_handles_for_keys(keys), loc='lower left',
              frameon=True, fontsize=BASE_FONT * 0.78, handlelength=1.3,
              labelspacing=0.45, borderpad=0.6)


def main():
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    draw_aggregate(ax)

    note = ('mean over the 7 toy envs; error bars: SEM of the mean over envs; '
            'faint: per-env means')
    fig.text(0.5, 0.005, note, ha='center', fontsize=BASE_FONT * 0.6, color='#666666')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    # figs/ = working copies; final_figures/ = the camera-ready set (per Jake
    # 2026-07-07) — kept current on every re-render.
    for d in (os.path.join(HERE, 'figs'),
              os.path.join(os.path.dirname(HERE), 'final_figures')):
        os.makedirs(d, exist_ok=True)
        for ext in ('pdf', 'png'):
            out = os.path.join(d, f'proto_pareto_7envs_aggregate.{ext}')
            fig.savefig(out, bbox_inches='tight', pad_inches=0.03,
                        **({'dpi': 150} if ext == 'png' else {}))
            print(f'wrote {out}')


if __name__ == '__main__':
    main()
