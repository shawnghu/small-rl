"""hf100 arms scatter v2 — the countdown_figure1 scatter + DEVELOPER-PICK stars.

Identical to countdown_figure1's left panel (same arms, means +- SEM, faint
per-seed markers, axis labels), plus one star per method: the single RUN a
GT-blind practitioner would deploy from that method's full hf100 candidate
pool, selected with developer-available metrics only —

    score = PROXY success (combined/ = own-test pass, the dev's "correct
    solution rate") - 2 x DETECTED hack (hack_freq_detectable, full dist);
    argmax over the pool (Jake 2026-07-13 — the soft RP2-objective analog,
    replacing the earlier hard monitored<=base filter). Star plotted at the
    winner's GROUND-TRUTH (hack, retain).

Pools: RP = all 18 RP runs (plain RP2/RP5 + RP{2,5} x lr{1,3} + anchor);
IP = both prompts x 3 seeds; PPS = 3 seeds; DN = 2 runs; GR = seeds 9/15/16
ONLY at the deployed fs0 (seed-matched to the other arms — GR's extra seeds
1-5 are excluded from both the arm mean and the star pool for fairness; Jake
2026-07-13). Every arm stars its best-scoring run. y-axis fixed to [0, 1].

v1 (hf100_arms_scatter.py) keeps the full per-config view (all RP/IP configs
separately); v2 is the condensed headline + selection story.

Run: .venv/bin/python figures_pareto/hf100_arms_scatter_v2.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from countdown_figure1 import OUT, draw_scatter, val  # noqa: E402

# arm label (matches the scatter legend) -> (color, [(glob, scale), ...])
STAR_POOLS = {
    "No intervention": ("#e0905a", [
        (f"{OUT}/countdown_code_rp2-0702-0026_fseval/"
         "countdown_code_hack_reward_penalty_amountmissing_s*.json", "1.0")]),
    "Reward penalty (dev-selected)": ("#d62728", [
        (f"{OUT}/countdown_hf100_rp_lccoh64_fseval/cdhf100_rp*_s*.json", "1.0"),
        (f"{OUT}/countdown_code_rp2-0702-0026_fseval/"
         "reward_penalty_countdown_code_hack_reward_penalty_amount*_s*.json", "1.0")]),
    "Inoculation prompting (best)": ("#a08070", [
        (f"{OUT}/countdown_hf100_ip_fseval/cdhf100_ip_*_s*.json", "1.0")]),
    "Preventative steering (best)": ("#8aa5a8", [
        (f"{OUT}/countdown_hf100_pps_fseval/cdhf100_pps_*_s*.json", "1.0")]),
    # seeds 9/15/16 only (this dir IS the 3-seed sweep; seeds5 dir excluded)
    "Gradient routing (ours)": ("#2ca02c", [
        (f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_*.json", "0.0")]),
}


def dev_pick(pool):
    """(run, gt_hack, retain, proxy, mon, score) of the developer-selected
    run: argmax of (proxy success - 2 x detected hack) over the pool."""
    cands = []
    for pat, scale in pool:
        for f in sorted(glob.glob(pat)):
            if "__" in os.path.basename(f):
                continue
            sm = json.load(open(f))["scales"]
            if scale not in sm:
                continue
            d = sm[scale]
            proxy = val(d, "combined/")
            mon = val(d, "hack_freq_detectable/")
            cands.append((os.path.basename(f).replace(".json", ""),
                          val(d, "hack_freq/"), val(d, "retain/"),
                          proxy, mon, proxy - 2.0 * mon))
    if not cands:
        return None, 0
    return max(cands, key=lambda c: c[5]), len(cands)


def main():
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14.45, 11.2))
    draw_scatter(ax, fs=1.15, gr_all_seeds=False)  # seed-matched GR (9/15/16)
    ax.set_ylim(0.0, 1.0)

    print("\n  developer picks (argmax proxy - 2 x detected):")
    for label, (color, pool) in STAR_POOLS.items():
        pick, n = dev_pick(pool)
        if pick is None:
            print(f"  [star] {label:32s} no candidates — no star")
            continue
        run, h, r, proxy, mon, score = pick
        print(f"  [star] {label:32s} {run}  score={score:.3f} "
              f"(proxy={proxy:.3f} mon={mon:.3f}) -> GT ({h:.3f}, {r:.3f})")
        ax.scatter(h, r, marker="*", s=2625, facecolor=color, edgecolor="black",
                   linewidth=1.6, zorder=6)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="*", linestyle="none",
                          markerfacecolor="white", markeredgecolor="black",
                          markeredgewidth=1.6, markersize=17,
                          label="Best seed by developer metrics"))
    ax.legend(handles=handles, loc="lower left", fontsize=20, framealpha=0.92)

    for d in (os.path.join(HERE, "figs"),
              os.path.join(os.path.dirname(HERE), "final_figures")):
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            out = os.path.join(d, f"hf100_arms_scatter_v2.{ext}")
            fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {d}/hf100_arms_scatter_v2.png/.pdf")


if __name__ == "__main__":
    main()
