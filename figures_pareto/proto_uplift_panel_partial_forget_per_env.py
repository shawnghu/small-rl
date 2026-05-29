"""Per-env version of proto_uplift_panel_partial_forget.

Same five classes, but plot one panel per env (no cross-env averaging).
Two figures produced — one for retain uplift, one for absolute hack rate.
2x4 layout with the legend slot.

Run:
    .venv/bin/python figures_pareto/proto_uplift_panel_partial_forget_per_env.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import ENVS, ENV_TITLE
from proto_uplift_panel_partial_forget import (
    CLASSES, _gather_traj, _gather_from_paths, _nocoh_run_dirs,
    _OPTIMA, ENV_MAX_STEPS,
)
from matplotlib.ticker import PercentFormatter
from proto_pareto_data import anchor_paths, no_intervention_paths
from proto_pareto_style_v2 import SLOT_ENVS, LEGEND_SLOT, BASE_FONT

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = BASE_FONT


def _per_env_curve(getter, env, family, subtract_base):
    """Returns (steps, mean_over_seeds, ci_over_seeds) for one env, one class.
    Reuses the getter's step_env_seeds dict but restricts to a single env."""
    step_env_seeds = getter(family, subtract_base)
    # step -> [per-seed values] for this env only
    by_step = {s: d.get(env, []) for s, d in step_env_seeds.items()
               if env in d and d[env]}
    if not by_step:
        return None, None, None
    steps = sorted(by_step)
    means, cis = [], []
    for s in steps:
        arr = np.array(by_step[s], dtype=float)
        means.append(float(arr.mean()))
        n = len(arr)
        if n > 1:
            cis.append(1.96 * float(arr.std(ddof=1) / np.sqrt(n)))
        else:
            cis.append(0.0)
    return np.array(steps), np.array(means), np.array(cis)


def _draw_env(ax, env, family, subtract_base, *, ylabel=False, xlabel=False):
    starts = []
    max_steps = ENV_MAX_STEPS.get(env, 2000)
    for label, color, _marker, getter in CLASSES:
        steps, mean, ci = _per_env_curve(getter, env, family, subtract_base)
        if steps is None or len(steps) == 0:
            continue
        progress = steps / max_steps
        ax.fill_between(progress, mean - ci, mean + ci, color=color,
                        alpha=0.18, linewidth=0)
        ax.plot(progress, mean, color=color, lw=2.0, label=label)
        starts.append(mean[0])
    base_y = 0.0 if subtract_base else (float(np.mean(starts)) if starts else 0.0)
    ax.axhline(base_y, color="0.35", lw=1.2, ls=(0, (6, 4)), zorder=1)
    ax.set_title(ENV_TITLE.get(env, env))
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    if not ylabel:
        ax.tick_params(labelleft=False)
    if not xlabel:
        ax.tick_params(labelbottom=False)


def _make_figure(family, subtract_base, ylabel, out_basename):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey=True)
    axes = axes.flatten()

    n_cols = 4
    for slot, env in SLOT_ENVS:
        ax = axes[slot]
        is_first_col = (slot % n_cols == 0)
        is_bottom_row = (slot >= n_cols)
        _draw_env(ax, env, family, subtract_base,
                  ylabel=is_first_col, xlabel=is_bottom_row)

    # Legend slot.
    lax = axes[LEGEND_SLOT]
    for s in lax.spines.values():
        s.set_visible(False)
    lax.set_xticks([]); lax.set_yticks([])
    handles = [Line2D([], [], color=c, lw=2.6, label=l)
               for l, c, _m, _g in CLASSES]
    lax.legend(handles=handles, loc="center", frameon=False,
               fontsize=BASE_FONT * 0.85, handlelength=2.0, labelspacing=0.9)

    # Shared y-label on the left column (top + bottom rows).
    axes[0].set_ylabel(ylabel)
    axes[n_cols].set_ylabel(ylabel)
    for slot, _env in SLOT_ENVS:
        if slot >= n_cols:
            axes[slot].set_xlabel("Training progress")

    fig.tight_layout(pad=0.4, h_pad=0.4, w_pad=0.4)
    out_pdf = os.path.join(HERE, "figs", out_basename + ".pdf")
    out_png = os.path.join(HERE, "figs", out_basename + ".png")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, dpi=130, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    plt.close(fig)


def main():
    # Per-env panels use ABSOLUTE retain (not uplift), so the y-axis aligns
    # with proto_pareto_7envs_gr_rp_v2's scatter. The cross-env uplift panel
    # uses subtract_base=True because envs differ in baseline scale — here
    # each panel is a single env so that concern doesn't apply.
    _make_figure("retain", subtract_base=False,
                 ylabel="Task performance (retain reward)",
                 out_basename="proto_uplift_per_env_retain_partial_forget")
    _make_figure("hack_freq", subtract_base=False,
                 ylabel="Reward hack rate",
                 out_basename="proto_uplift_per_env_hackfreq_partial_forget")


if __name__ == "__main__":
    main()
