"""Composite Figure-1-style figure for the countdown_code env, hf50 variant
(50/50 hackable contract prompts; replaces the old 100%-hackable coh64 data).

Left:  the hf50 arms endpoint scatter (hf50_arms_scatter.draw_scatter,
       no-stars version, per Jake 2026-07-09): GT hack over all prompts vs
       retain, mean +/- SEM per arm with faint per-seed markers. Arms:
       do-nothing / RP=2 / RP=5 / GR-nocoh lr/3 at the per-seed
       classifier-picked forget scale / frozen base.
Right: adapter decomposition of the GRAFT runs ONLY — the same 8 runs shown as
       pre-/post-ablation on the left, evaluated in three adapter configs:
       both adapters (trained config), retain adapter only (deployed config),
       and forget adapter only. The forget-only series is DARK RED and DOTTED
       because it is a diagnostic probe, never a trained/deployed config.
       Top: task-performance UPLIFT — per run, retain(t) minus retain at that
       run's first eval row (step 12); same convention as the 7-env Figure 1
       (proto_uplift_panel_v1.class_curve subtract_base=True). Dashed gray
       reference at 0. Bottom: ABSOLUTE hackable-slice hack rate; dashed gray
       reference at the mean of the three series' first-eval values.

Arms (all 200-step Modal runs, evals every 12 steps through 192):
  - No intervention      countdown_hf50_dn            seeds 9/15/16
  - Reward Penalty p=2   countdown_hf50_rp2           seeds 9/15/16
                         (best mean RP on hf50)
  - GRAFT                countdown_hf50_gr_nocoh_lr3  seeds 1-5/9/15/16
                         (no-coherence, lr/3; 8 seeds)

Unified colors (Jake's scheme): RP red #d62728, GR deployed green #2ca02c,
GR both-adapters blue #1f77b4, forget-only dark red #8b0000 dotted,
no-intervention orange #ff7f0e.

Data quirks handled here:
  - Modal-restart duplicate eval rows: dedup by step, keep LAST occurrence
    (each routing_eval row carries all three modes, so per-step dedup is
    per-(step, mode) dedup).

Run: .venv/bin/python figures_pareto/countdown_figure1.py
"""
import json
import os
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")

import sys
sys.path.insert(0, HERE)
from hf50_arms_scatter import draw_scatter  # noqa: E402  (left panel)

GR_SEEDS = (1, 2, 3, 4, 5, 9, 15, 16)
BASE_SEEDS = (9, 15, 16)

GR_TMPL = f"{OUT}/countdown_hf50_gr_nocoh_lr3/cdhf50_gr_nocoh_lr3_s{{s}}"
RP_TMPL = f"{OUT}/countdown_hf50_rp2/cdhf50_rp2_s{{s}}"
DN_TMPL = f"{OUT}/countdown_hf50_dn/cdhf50_dn_s{{s}}"

FSEVAL_TMPL = (f"{OUT}/countdown_hf50_gr_nocoh_lr3_fseval"
               f"/cdhf50_gr_nocoh_lr3_s{{s}}.json")
# Per-seed classifier-picked forget scales (keys into fseval 'scales').
FS_PICKS = {1: "0.3", 2: "0.5", 3: "0.4", 4: "0.6", 5: "0.4",
            9: "0.3", 15: "0.4", 16: "0.3"}

# Hackable-slice hack channel — verified present in every arm's eval rows.
HACK_SLUG = "hack_freq_hackable"

# Right panel: adapter configs of the GR runs. (mode, color, ls, lw, label);
# forget_only is dark red + dotted = diagnostic, never trained/deployed.
GR_MODES = [
    ("both", "#1f77b4", "-", 2.4, "Both adapters"),
    ("retain_only", "#2ca02c", "-", 2.4, "Retain adapter only (deployed)"),
    ("forget_only", "#8b0000", ":", 2.6, "Forget adapter only (diagnostic)"),
]

plt.rcParams["font.size"] = 20
plt.rcParams["axes.unicode_minus"] = False


def load_rows(run_dir):
    """routing_eval rows, deduped by step keeping the LAST occurrence
    (Modal container restarts append a second copy of earlier steps). Each
    row carries all three modes' channels, so this is a (step, mode) dedup."""
    path = os.path.join(run_dir, "routing_eval.jsonl")
    by_step = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


def chan(row, mode, slug):
    """Exact-segment channel lookup: keys are '<mode>/<slug>/<detector...>'.
    Prefix matching on 'hack_freq' would also hit hack_freq_hackable etc."""
    for k, v in row.items():
        parts = k.split("/", 2)
        if len(parts) >= 2 and parts[0] == mode and parts[1] == slug:
            return float(v)
    raise KeyError(f"{mode}/{slug} not in eval row (step {row.get('step')})")


def series(run_dir, mode, slug):
    """[(step, value)] for one channel of one run."""
    return [(r["step"], chan(r, mode, slug)) for r in load_rows(run_dir)]


def sem(xs):
    return st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


# ---------------- shared curve drawing ----------------
def seed_curve(ax, dirs, mode, slug, color, ls, lw, label, subtract_base):
    """Seed-mean curve +/- SEM band; per-seed traces faint (keeping the series
    linestyle, so forget_only's faint traces are dotted too).
    subtract_base=True -> per run, value(t) - value(first eval row) (uplift,
    matching the 7-env Figure 1 convention).
    Returns (last-step per-seed values, first-eval seed-mean)."""
    steps_ref, per_seed = None, []
    for d in dirs:
        sv = series(d, mode, slug)
        steps = [p[0] for p in sv]
        if steps_ref is None:
            steps_ref = steps
        assert steps == steps_ref, \
            f"eval-step mismatch across seeds ({mode}/{slug}, {d})"
        base = sv[0][1] if subtract_base else 0.0
        per_seed.append([v - base for _, v in sv])
    for vals in per_seed:
        ax.plot(steps_ref, vals, color=color, ls=ls, lw=0.7, alpha=0.22,
                zorder=2)
    arr = np.array(per_seed)
    mean = arr.mean(axis=0)
    band = np.array([sem(list(col)) for col in arr.T])
    ax.fill_between(steps_ref, mean - band, mean + band, color=color,
                    alpha=0.15, zorder=3, linewidth=0)
    ax.plot(steps_ref, mean, color=color, ls=ls, lw=lw, zorder=4, label=label)
    return list(arr[:, -1]), float(mean[0])


# ---------------- right: GR adapter decomposition curves ----------------
def draw_adapter_panels(ax_top, ax_bot):
    gr_dirs = [GR_TMPL.format(s=s) for s in GR_SEEDS]
    for mode, color, ls, lw, label in GR_MODES:
        seed_curve(ax_top, gr_dirs, mode, "retain", color, ls, lw, label,
                   subtract_base=True)
    ax_top.axhline(0.0, color="0.35", lw=1.8, ls=(0, (6, 4)), zorder=1)
    ax_top.set_ylabel("Task performance\nimprovement")

    starts = [seed_curve(ax_bot, gr_dirs, mode, HACK_SLUG, color, ls, lw,
                         label, subtract_base=False)[1]
              for mode, color, ls, lw, label in GR_MODES]
    base_y = float(np.mean(starts))
    ax_bot.axhline(base_y, color="0.35", lw=1.8, ls=(0, (6, 4)), zorder=1)
    ax_bot.text(0.985, base_y + 0.02, "initial level",
                transform=ax_bot.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=12, color="0.35")
    ax_bot.set_ylabel("Reward hack rate\n(hackable prompts)")
    # Same quantity as the left-bottom panel -> same unit (percent). The top
    # panel stays fractional: it is a performance delta, not a rate.
    ax_bot.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_bot.set_xlabel("Training step")


def main():
    fig = plt.figure(figsize=(17.0, 9.0))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[8.6, 8.4], wspace=0.0)
    TOP, BOT = 0.97, 0.10

    axl = sub_l.subplots(1, 1)
    draw_scatter(axl, with_stars=False, fs=1.55, title=False)
    sub_l.subplots_adjust(left=0.13, right=0.97, top=TOP, bottom=BOT)

    axr_top, axr_bot = sub_r.subplots(2, 1, sharex=True)
    draw_adapter_panels(axr_top, axr_bot)
    # Legend in the bottom panel's upper left: retain_only stays low there and
    # the both/forget curves only climb on the right half.
    axr_bot.legend(loc="upper left", frameon=True, fontsize=14)
    # left=0.17: keeps the rotated ylabels clear of the left subfigure's
    # panel frame (they sit flush against it at 0.15).
    sub_r.subplots_adjust(left=0.17, right=0.97, top=TOP, bottom=BOT,
                          hspace=0.07)
    sub_r.align_ylabels([axr_top, axr_bot])

    for ax in (axr_top, axr_bot):
        ax.set_xlim(0, 200)
        ax.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    axl.set_axisbelow(True)

    # figs/ = working copies; final_figures/ = the camera-ready set (per Jake
    # 2026-07-07) — kept current on every re-render.
    for d in (os.path.join(HERE, "figs"), os.path.join(ROOT, "final_figures")):
        os.makedirs(d, exist_ok=True)
        for ext in ("pdf", "png"):
            out = os.path.join(d, f"countdown_figure1.{ext}")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04,
                        **({"dpi": 150} if ext == "png" else {}))
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
