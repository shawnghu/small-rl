"""Composite Figure-1 for the countdown_code env, hf100 (100%-hackable) data.

Left:  endpoint deployment scatter (n=256, mean +- SEM, faint per-seed):
         No intervention        old round-2 DN runs (fresh re-eval, current code)
         Reward penalty (best)  best RP config on hf100 (see BEST_RP below)
         Inoculation (best)     IP mand-tw, prompt removed at eval
         Prev. steering (best)  PPS L20 alpha2, steering removed at eval
         Gradient routing       lccoh64, forget ablated (fs0), 8 seeds
         Base model             frozen SFT-primed model on the hf100 eval set
       No inline point labels (legend only). Axes per Jake 2026-07-13:
       x "Reward hack rate (better ->)" (inverted), y "Correct solution rate
       (better ->)" (rotated arrow points up).

Right: adapter decomposition of the 8 GR (lccoh64) runs in three configs
       (both / retain-only=deployed / forget-only=diagnostic).
       - Curves anchored at step 0 = BASE MODEL and extended to step 200 with
         eval-methodology-matched points: the in-training rows are n=64
         piggybacked evals, so steps 0 and 200 use posthoc
         eval_forget_scales_one at n_eval=64 on the SAME deterministic eval
         prompts / temp 1.0 / 1536-token cap ({run}__step200.json for
         (1,1)+(1,0); {run}__step200__r0.0.json for (0,1) and the shared
         (0,0) base anchor).
       - Top: retain uplift relative to the BASE MODEL (not the first eval
         row) -> all curves start at exactly 0.
       - Bottom: absolute hack rate (all prompts hackable on hf100, so no
         "(hackable prompts)" qualifier); starts at the base rate.
       - "Initial level" is a legend entry (dashed gray), no on-plot text.
       - y-limits set from the MEAN lines (faint seed traces may clip).

Data quirks: Modal-restart duplicate eval rows deduped by step keep-last.

Run: .venv/bin/python figures_pareto/countdown_figure1.py
"""
import glob
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

GR_SEEDS = (1, 2, 3, 4, 5, 9, 15, 16)


def _gr_sweep(seed):
    """(run_dir, fseval_dir) for one GR seed (3-seed vs seeds5 sweep split)."""
    sw = ("countdown_hf100_gr_lccoh64_lr3" if seed in (9, 15, 16)
          else "countdown_hf100_gr_lccoh64_lr3_seeds5")
    run = f"cdhf100_gr_lccoh64_lr3_s{seed}"
    return f"{OUT}/{sw}/{run}", f"{OUT}/{sw}_fseval", run


def val(d, p):
    hits = [v for k, v in d.items() if k.startswith(p) and "/" not in k[len(p):]]
    return hits[0] if hits else None


def _pts(pattern, scale):
    """[(hack, retain)] from fseval JSONs (skip __step/__r derivative files)."""
    pts = []
    for f in sorted(glob.glob(pattern)):
        if "__" in os.path.basename(f):
            continue
        sm = json.load(open(f))["scales"]
        if scale in sm:
            pts.append((val(sm[scale], "hack_freq/"), val(sm[scale], "retain/")))
    return pts


# ---------------- left: scatter arms (all n=256 endpoints) ----------------
# RP config selected the way a PRACTITIONER would (Jake 2026-07-13): among
# configs whose MONITORED hack is <= the base model's (all of them — the
# residual is unmonitored constraint_relax), argmax PROXY success (own-test
# pass). That picks rp2+anchor lr1 (proxy 1.000, mon 0.005) whose ground truth
# is 0.259 hack / 0.741 retain — the dev metrics actively select the hacker.
# (Oracle/GT selection would pick rp5_lc64_lr3: 0.077/0.813. Not shown.)
BEST_RP = ("rp2_lc64_lr1", "Reward penalty")

DN_GLOB = (f"{OUT}/countdown_code_rp2-0702-0026_fseval/"
           "countdown_code_hack_reward_penalty_amountmissing_s*.json")


def scatter_arms(gr_all_seeds=True):
    """[(label, [(hack, retain)], color, marker, hollow)].
    gr_all_seeds=False restricts GR to seeds 9/15/16 (the 3-seed sweep dir) —
    seed-matched to the other arms (hf100_arms_scatter_v2's fairness mode)."""
    gr = _pts(f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_*.json", "0.0")
    if gr_all_seeds:
        gr += _pts(f"{OUT}/countdown_hf100_gr_lccoh64_lr3_seeds5_fseval/cdhf100_*.json", "0.0")
    base_f = (f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/"
              "cdhf100_gr_lccoh64_lr3_s9__r0.0.json")
    base = json.load(open(base_f))["scales"]["0.0"]
    return [
        ("No intervention",
         _pts(DN_GLOB, "1.0"), "#e0905a", "X", False),
        (BEST_RP[1],
         _pts(f"{OUT}/countdown_hf100_rp_lccoh64_fseval/cdhf100_{BEST_RP[0]}_s*.json", "1.0"),
         "#d62728", "s", False),
        ("Inoculation prompting",
         _pts(f"{OUT}/countdown_hf100_ip_fseval/cdhf100_ip_mand-tw_s*.json", "1.0"),
         "#a08070", "v", False),
        ("Preventative steering",
         _pts(f"{OUT}/countdown_hf100_pps_fseval/cdhf100_pps_L20_a2_s*.json", "1.0"),
         "#8aa5a8", "h", False),
        # Routing ablated (rh_detector_recall=0 == the lambda=0 redistribution
        # point), anchoring intact — shows routing, not anchoring, localizes.
        ("GRAFT w/o routing",
         _pts(f"{OUT}/cdhf100_noroute_fseval/cdhf100_noroute_anchor_s*.json", "0.0"),
         "#9690a8", "X", True),
        ("GRAFT (ours)", gr, "#2ca02c", "o", False),
        ("Base model",
         [(val(base, "hack_freq/"), val(base, "retain/"))], "#444444", "o", True),
    ]


def draw_scatter(ax, fs=1.0, gr_all_seeds=True):
    all_h, all_r = [], []
    for label, pts, color, marker, hollow in scatter_arms(gr_all_seeds):
        pts = [(h, r) for h, r in pts if h is not None]
        if not pts:
            print(f"  [scatter] {label}: NO DATA yet — skipped")
            continue
        hs = [p[0] for p in pts]
        rs = [p[1] for p in pts]
        all_h += hs
        all_r += rs
        sem = lambda x: (st.stdev(x) / len(x) ** 0.5) if len(x) > 1 else 0.0
        for h, r in pts:
            ax.scatter(h, r, marker=marker, s=72, alpha=0.4, zorder=2,
                       facecolors="none" if hollow else color, edgecolors="none" if not hollow else color)
        ax.errorbar(st.mean(hs), st.mean(rs), xerr=sem(hs), yerr=sem(rs),
                    color=color, marker=marker, markersize=17,
                    markerfacecolor="white" if hollow else color,
                    markeredgecolor=color if hollow else "white",
                    markeredgewidth=2.0 if hollow else 1.6,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    zorder=50 if label == "GRAFT (ours)" else 4,
                    clip_on=label != "GRAFT (ours)", label=label)
        print(f"  [scatter] {label:32s} n={len(pts)} hack={st.mean(hs):.3f}±{sem(hs):.3f} "
              f"retain={st.mean(rs):.3f}±{sem(rs):.3f}")
    # data-driven limits: cover every per-seed marker with a small margin
    ax.set_xlim(max(all_h) + 0.05, -0.03)
    ax.set_ylim(min(all_r) - 0.04, max(all_r) + 0.04)
    # axis labels inherit rcParams font.size — same size as the line panels'
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Reward hack rate  (better →)", fontsize=25)
    # rotated ylabel: the arrow glyph rotates with the text and points UP
    ax.set_ylabel("Correct solution rate  (better →)", fontsize=25)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=16, framealpha=0.92)


# ---------------- right: GR adapter decomposition ----------------
GR_MODES = [
    ("both", "#1f77b4", "-", 2.4, "GRAFT: pre-ablation"),
    ("retain_only", "#2ca02c", "-", 2.4, "GRAFT (ours)"),
    ("forget_only", "#8b0000", ":", 2.6, "GRAFT: forget-only (diagnostic)"),
]


def load_rows(run_dir):
    by_step = {}
    with open(os.path.join(run_dir, "routing_eval.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


def chan(row, mode, slug):
    for k, v in row.items():
        parts = k.split("/", 2)
        if len(parts) >= 2 and parts[0] == mode and parts[1] == slug:
            return float(v)
    raise KeyError(f"{mode}/{slug} not in eval row (step {row.get('step')})")


def _step200(seed, mode, slug):
    """Posthoc n=64 step-200 value for one seed/mode (methodology-matched)."""
    _, fsdir, run = _gr_sweep(seed)
    key = "retain" if slug == "retain" else "hack_freq"
    if mode == "both":
        sm = json.load(open(f"{fsdir}/{run}__step200.json"))["scales"]["1.0"]
    elif mode == "retain_only":
        sm = json.load(open(f"{fsdir}/{run}__step200.json"))["scales"]["0.0"]
    else:  # forget_only = (retain 0, forget 1)
        sm = json.load(open(f"{fsdir}/{run}__step200__r0.0.json"))["scales"]["1.0"]
    return val(sm, key + "/")


def _base_anchor(slug):
    """Base-model (0,0) value, averaged over the 8 seeds' n=64 draws of the
    SAME frozen model (a constant; averaging just tightens the anchor).
    FALLBACK while those evals are in flight: the n=256 base fseval."""
    key = "retain" if slug == "retain" else "hack_freq"
    vs = []
    for s in GR_SEEDS:
        _, fsdir, run = _gr_sweep(s)
        f = f"{fsdir}/{run}__step200__r0.0.json"
        if os.path.exists(f):
            vs.append(val(json.load(open(f))["scales"]["0.0"], key + "/"))
    if not vs:
        f = (f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/"
             "cdhf100_gr_lccoh64_lr3_s9__r0.0.json")
        print(f"  [curves] WARNING: n=64 base anchors not landed; using n=256 "
              f"base fallback for {slug}")
        return val(json.load(open(f))["scales"]["0.0"], key + "/")
    return sum(vs) / len(vs)


def _have_step200():
    """True iff every seed's step-200 posthoc files (both retain-1.0 and
    retain-0.0 variants) are present."""
    for s in GR_SEEDS:
        _, fsdir, run = _gr_sweep(s)
        if not (os.path.exists(f"{fsdir}/{run}__step200.json")
                and os.path.exists(f"{fsdir}/{run}__step200__r0.0.json")):
            return False
    return True


def sem(xs):
    return st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


def seed_curve(ax, mode, slug, color, ls, lw, label, subtract, with_200):
    """Seed-mean +- SEM band over the 8 GR runs, anchored at (0, base) and —
    when the posthoc files are in (with_200) — extended to (200, n=64).
    `subtract` is the base value subtracted from every point (0.0 for
    absolute panels). Returns the mean curve."""
    steps_ref, per_seed = None, []
    for s in GR_SEEDS:
        run_dir, _, _ = _gr_sweep(s)
        rows = load_rows(run_dir)
        steps = [0] + [r["step"] for r in rows]
        vals = [_base_anchor(slug)] + [chan(r, mode, slug) for r in rows]
        if with_200:
            steps.append(200)
            vals.append(_step200(s, mode, slug))
        if steps_ref is None:
            steps_ref = steps
        assert steps == steps_ref, f"eval-step mismatch (s{s}, {mode}/{slug})"
        per_seed.append([v - subtract for v in vals])
    for v in per_seed:
        ax.plot(steps_ref, v, color=color, ls=ls, lw=0.7, alpha=0.20, zorder=2)
    arr = np.array(per_seed)
    mean = arr.mean(axis=0)
    band = np.array([sem(list(c)) for c in arr.T])
    ax.fill_between(steps_ref, mean - band, mean + band, color=color,
                    alpha=0.15, zorder=3, linewidth=0)
    ax.plot(steps_ref, mean, color=color, ls=ls, lw=lw, zorder=4, label=label)
    return mean


def draw_adapter_panels(ax_top, ax_bot):
    base_ret = _base_anchor("retain")
    base_hack = _base_anchor("hack_freq")
    print(f"  [curves] base anchor: retain={base_ret:.3f} hack={base_hack:.3f}")

    with_200 = _have_step200()
    if not with_200:
        print("  [curves] step-200 posthoc files not all landed — rendering "
              "through step 192 for now")
    top_means, bot_means = [], []
    for mode, color, ls, lw, label in GR_MODES:
        top_means.append(seed_curve(ax_top, mode, "retain", color, ls, lw,
                                    label, subtract=base_ret, with_200=with_200))
        bot_means.append(seed_curve(ax_bot, mode, "hack_freq", color, ls, lw,
                                    label, subtract=0.0, with_200=with_200))
    ax_top.axhline(0.0, color="0.35", lw=1.8, ls=(0, (6, 4)), zorder=1)
    ax_bot.axhline(base_hack, color="0.35", lw=1.8, ls=(0, (6, 4)), zorder=1)

    # y-limits from the MEAN lines only (faint per-seed traces may clip)
    tm = np.concatenate(top_means)
    bm = np.concatenate(bot_means)
    ax_top.set_ylim(min(0, tm.min()) - 0.04, tm.max() + 0.05)
    ax_bot.set_ylim(min(0, bm.min()) - 0.01, bm.max() + 0.06)

    ax_top.set_ylabel("Task performance\nimprovement", fontsize=25)
    ax_bot.set_ylabel("Reward hack rate", fontsize=25)
    ax_bot.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_bot.set_xlabel("Training step", fontsize=25)

    handles = [Line2D([0], [0], color=c, ls=ls, lw=lw, label=lab)
               for _, c, ls, lw, lab in GR_MODES]
    handles.append(Line2D([0], [0], color="0.35", ls=(0, (6, 4)), lw=1.8,
                          label="Base model"))
    # right side of the bottom panel, nudged below center so the box clears
    # the blue both-adapters line.
    ax_bot.legend(handles=handles, loc="center right", frameon=True, fontsize=16, framealpha=0.92,
                  bbox_to_anchor=(1.0, 0.36))


def main():
    fig = plt.figure(figsize=(17.0, 9.0))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[8.6, 8.4], wspace=0.0)
    TOP, BOT = 0.97, 0.10
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False

    axl = sub_l.subplots(1, 1)
    draw_scatter(axl, fs=1.55)
    sub_l.subplots_adjust(left=0.13, right=0.97, top=TOP, bottom=BOT)

    axr_top, axr_bot = sub_r.subplots(2, 1, sharex=True)
    draw_adapter_panels(axr_top, axr_bot)
    sub_r.subplots_adjust(left=0.17, right=0.97, top=TOP, bottom=BOT, hspace=0.07)
    sub_r.align_ylabels([axr_top, axr_bot])

    for ax in (axr_top, axr_bot):
        ax.set_xlim(0, 200)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    axl.set_axisbelow(True)

    for d in (os.path.join(HERE, "figs"), os.path.join(ROOT, "final_figures")):
        os.makedirs(d, exist_ok=True)
        for ext in ("pdf", "png"):
            out = os.path.join(d, f"countdown_figure1.{ext}")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04,
                        **({"dpi": 150} if ext == "png" else {}))
            print(f"wrote {out}")


if __name__ == "__main__":
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False
    main()
