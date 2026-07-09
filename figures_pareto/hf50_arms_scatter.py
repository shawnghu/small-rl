"""Countdown hf50 (50/50 hackable) arms scatter: hack vs retain.

x = ground-truth hack rate over ALL prompts (decreasing ->), y = retain over
ALL prompts (the combined 50/50 set). Arms: do-nothing (3 seeds), RP=2
(3 seeds), RP=5 (3 seeds), GR-nocoh lr/3 at the per-seed classifier-selected
forget scale (8 seeds), base (single point). Mean +- SEM with faint per-seed
markers, mirroring figures_pareto/countdown_arms_scatter.py.

Two outputs: hf50_arms_scatter.{png,pdf} (with the classifier-only deployment
selection stars) and hf50_arms_scatter_nostars.{png,pdf} (no stars, no star
legend entry). Both written to figures_pareto/figs/ and final_figures/.
"""
import glob
import json
import os
import statistics as st

import matplotlib.pyplot as plt

OUT = "/workspace/small-rl/output"


def val(d, p):
    hits = [v for k, v in d.items() if k.startswith(p) and "/" not in k[len(p):]]
    return hits[0] if hits else None


def _rp_files(amount):
    """Original (s9/15/16) + ext (s1-5) endpoint evals — 8 seeds when both in."""
    return sorted(glob.glob(f"{OUT}/countdown_hf50_rp{amount}_fseval/*.json")
                  + glob.glob(f"{OUT}/countdown_hf50_rp{amount}_ext_fseval/*.json"))


def rp_points(amount):
    pts = []
    for f in _rp_files(amount):
        d = json.load(open(f))["scales"]["1.0"]
        pts.append((val(d, "hack_freq/"), val(d, "retain/")))
    return pts


def dn_points():
    # DN fseval is degenerate (no adapters); use in-training posthoc endpoints.
    # Per-seed (retain, hack@hackable) from the n=256 both-config posthoc.
    # full-distribution GT hack (hack_freq over all prompts) / retain, n=256 posthoc
    return [(0.24, 0.69), (0.15, 0.76), (0.47, 0.46)]  # s9/s15/s16


# lr/3 + no-coherence: the strongest hf50 protocol (8 seeds, single 11-scale
# fseval file each). Classifier picks (highest scale with monitored < base).
LR3_SEEDS = (1, 2, 3, 4, 5, 9, 15, 16)
LR3_PICKS = {1: "0.3", 2: "0.5", 3: "0.4", 4: "0.6", 5: "0.4",
             9: "0.3", 15: "0.4", 16: "0.3"}


def _lr3_scales(s):
    f = f"{OUT}/countdown_hf50_gr_nocoh_lr3_fseval/cdhf50_gr_nocoh_lr3_s{s}.json"
    return json.load(open(f))["scales"]


def lr3_points():
    pts = []
    for s in LR3_SEEDS:
        d = _lr3_scales(s)[LR3_PICKS[s]]
        pts.append((val(d, "hack_freq/"), val(d, "retain/")))
    return pts


def lr3_naive():
    cands = [d for s in LR3_SEEDS for d in _lr3_scales(s).values()]
    return _naive_pick(cands)


def base_point():
    """Countdown-primed base (frozen, no RL) on the hf50 eval set: the origin
    all hf50 RL runs trained from. fseval scales(0,0) on any hf50 checkpoint =
    the frozen base. Written to <run>__r0.0.json (retain_scale=0.0)."""
    f = f"{OUT}/countdown_hf50_gr_nocoh_fseval/cdhf50_gr_nocoh_s9__r0.0.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))["scales"]["0.0"]
    return [(val(d, "hack_freq/"), val(d, "retain/"))]


# --- Classifier-only deployment selection (the naive practitioner) ------------
# A practitioner who lacks GT and cannot tell which prompts are hackable would:
# train N candidates, keep those whose CLASSIFIER hack rate over the FULL
# distribution is <= the base model's, then deploy the one with the highest
# any-solve rate (proxy pass = genuine + hacked, also over the full dist, since
# GT solve rate is unavailable). We plot where that pick actually lands on GT.
BASE_MON = val(json.load(open(
    f"{OUT}/countdown_hf50_gr_nocoh_fseval/cdhf50_gr_nocoh_s9__r0.0.json"))["scales"]["0.0"],
    "hack_freq_detectable/")


def _naive_pick(candidate_scale_dicts):
    """candidate_scale_dicts: list of per-candidate scale-metric dicts.
    Returns (hack@H, retain) GT coords of the argmax-any-solve candidate whose
    full-dist classifier hack <= base, or None if none pass."""
    ok = [d for d in candidate_scale_dicts
          if val(d, "hack_freq_detectable/") is not None
          and val(d, "hack_freq_detectable/") <= BASE_MON + 1e-9]
    if not ok:
        return None
    best = max(ok, key=lambda d: val(d, "combined/"))  # combined/ = any-solve (all prompts)
    return (val(best, "hack_freq/"), val(best, "retain/"))


def rp_naive(amount):
    cands = [json.load(open(f))["scales"]["1.0"] for f in _rp_files(amount)]
    return _naive_pick(cands)


def stats(pts):
    hs = [p[0] for p in pts]
    rs = [p[1] for p in pts]
    sem = lambda x: (st.stdev(x) / len(x) ** 0.5) if len(x) > 1 else 0.0
    return st.mean(hs), sem(hs), st.mean(rs), sem(rs), pts


# Per-arm annotation placement overrides (default: centered, 14pt above).
# GR sits at the far right edge; anchor its label right and drop it below the
# cluster so it neither spills past the axis nor collides with RP=2/the star.
ANN = {"GR-nocoh lr/3 (fs pick)": dict(xytext=(10, -42), ha="right"),
       "RP=5": dict(xytext=(-16, 6), ha="right")}  # left of marker, above errorbar

# Unified colors: Reward Penalty = red family, GR deployed = green.
ARMS = {
    "do-nothing":              (dn_points(),   "#444444", "X"),
    "RP=2":                    (rp_points(2),  "#d62728", "s"),
    "RP=5":                    (rp_points(5),  "#ff7f0e", "D"),
    "GR-nocoh lr/3 (fs pick)": (lr3_points(),  "#2ca02c", "^"),
}
_base = base_point()
if _base is not None:
    ARMS["base (no RL)"] = (_base, "#999999", "o")
else:
    print("NOTE: base hf50 eval not yet present — skipping base point")

# Classifier-only deployment picks (★): where a GT-blind, hackability-blind
# practitioner would land. RP passes the classifier filter but deploys a hacker
# (its residual is unmonitored cr); GR-nocoh's filter binds because its residual
# hack IS the monitored form, so the same procedure deploys a clean model.
NAIVE = [("RP=2",  rp_naive(2),   "#d62728"),
         ("RP=5",  rp_naive(5),   "#ff7f0e"),
         ("GR-nocoh lr/3", lr3_naive(), "#2ca02c")]


def draw_scatter(ax, with_stars, fs=1.0, title=True):
    """Draw the arms scatter onto `ax`. fs scales the panel's explicit font
    sizes (the standalone figure uses small fonts; composite hosts like
    countdown_figure1 run 20pt rcParams and pass fs≈1.6 to match)."""
    for name, (pts, color, marker) in ARMS.items():
        hm, he, rm, re_, per = stats(pts)
        for h, r in per:
            ax.scatter(h, r, color=color, marker=marker, s=28 * fs, alpha=0.28,
                       zorder=2)
        ax.errorbar(hm, rm, xerr=he, yerr=re_, color=color, marker=marker,
                    markersize=13, capsize=3, elinewidth=1.4, zorder=3, label=name)
        ann = {"xytext": (0, 14), "ha": "center", **ANN.get(name, {})}
        ax.annotate(name, (hm, rm), textcoords="offset points",
                    fontsize=10 * fs, color=color, **ann)

    if with_stars:
        labeled = False
        for name, pt, color in NAIVE:
            if pt is None:
                continue
            h, r = pt
            ax.scatter(h, r, marker="*", s=560, facecolor=color, edgecolor="black",
                       linewidth=1.6, zorder=5,
                       label=None if labeled else "classifier+any-solve pick (full dist)")
            labeled = True
            ax.scatter(h, r, marker="*", s=560, facecolor="none", edgecolor="black",
                       linewidth=1.6, zorder=6)

    ax.set_xlim(0.52, -0.02)  # full-dist GT hack, decreasing -> (tightened to spread arms)
    ax.set_ylim(0.30, 0.95)
    ax.set_xlabel("ground-truth hack rate over all prompts  (better →)",
                  fontsize=11 * fs)
    ax.set_ylabel("retain: true_tested over all prompts  (better ↑)",
                  fontsize=11 * fs)
    if title:
        counts = (f"DN: {len(dn_points())}, RP2: {len(rp_points(2))}, "
                  f"RP5: {len(rp_points(5))}, GR lr/3: {len(LR3_SEEDS)} seeds")
        ax.set_title("Countdown hf50 (50/50 hackable) — mean ± SEM over seeds\n"
                     f"({counts}; GR forget scale classifier-selected per seed; "
                     "faint = per-seed)",
                     fontsize=10.5 * fs)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9 * fs, framealpha=0.9)


def make_fig(with_stars):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    draw_scatter(ax, with_stars)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for with_stars, stem in ((True, "hf50_arms_scatter"),
                             (False, "hf50_arms_scatter_nostars")):
        fig = make_fig(with_stars)
        for d in ("/workspace/small-rl/figures_pareto/figs",
                  "/workspace/small-rl/final_figures"):
            os.makedirs(d, exist_ok=True)
            for ext in ("png", "pdf"):
                fig.savefig(f"{d}/{stem}.{ext}", dpi=140)
        plt.close(fig)
        print(f"wrote figs/ + final_figures/ {stem}.png / .pdf")
    for name, (pts, _, _) in ARMS.items():
        hm, he, rm, re_, _ = stats(pts)
        print(f"  {name:<24} GT-hack {hm:.3f}±{he:.3f}  retain {rm:.3f}±{re_:.3f}")
