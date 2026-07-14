"""Countdown hf100 (100%-hackable) arms scatter: hack vs retain.

The harder env where hf50's forgiving clean-corner collapses: only GR holds;
the single-policy interventions each fail in their characteristic way. x =
ground-truth hack rate over all prompts (decreasing ->), y = retain
(true_tested). Mean +- SEM with faint per-seed markers.

Arms (all deployed / endpoint, n=256):
  - GR lccoh64          fs0 (forget ablated), 8 seeds        -> holds
  - RP+anchor RP{2,5} x lr{1,3}, scale 1.0, 3 seeds each     -> unmonitored substitution
  - IP mand-tw / mand-tw-norm, scale 1.0, 3 seeds each       -> gate erodes
  - PPS L20 a2, scale 1.0 (steering removed), 3 seeds        -> internalizes hack
  - base (frozen SFT-primed model on hf100 prompts)          -> origin

Run: .venv/bin/python figures_pareto/hf100_arms_scatter.py
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


def _load(pattern, scale, retain0=False):
    """[(hack, retain)] over fseval JSONs matching pattern, at `scale`. Skips
    __r* files (retain-scale base evals live in the GR dir but aren't seeds)."""
    pts = []
    for f in sorted(glob.glob(pattern)):
        if "__r" in os.path.basename(f):
            continue
        sc = json.load(open(f))["scales"]
        if scale not in sc:
            continue
        d = sc[scale]
        pts.append((val(d, "hack_freq/"), val(d, "retain/")))
    return pts


def gr_points():
    return (_load(f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/*.json", "0.0")
            + _load(f"{OUT}/countdown_hf100_gr_lccoh64_lr3_seeds5_fseval/*.json", "0.0"))


def rp_points(cfg):
    return _load(f"{OUT}/countdown_hf100_rp_lccoh64_fseval/cdhf100_{cfg}_s*.json", "1.0")


def ip_points(name):
    return _load(f"{OUT}/countdown_hf100_ip_fseval/cdhf100_ip_{name}_s*.json", "1.0")


def pps_points():
    return _load(f"{OUT}/countdown_hf100_pps_fseval/cdhf100_pps_L20_a2_s*.json", "1.0")


def base_point():
    # frozen model = fseval retain_scale 0.0 on an hf100 GR checkpoint (__r0.0.json)
    f = f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_gr_lccoh64_lr3_s9__r0.0.json"
    if not os.path.exists(f):
        return []
    d = json.load(open(f))["scales"]["0.0"]
    return [(val(d, "hack_freq/"), val(d, "retain/"))]


def stats(pts):
    hs = [p[0] for p in pts if p[0] is not None]
    rs = [p[1] for p in pts if p[1] is not None]
    if not hs:
        return None
    sem = lambda x: (st.stdev(x) / len(x) ** 0.5) if len(x) > 1 else 0.0
    return st.mean(hs), sem(hs), st.mean(rs), sem(rs), list(zip(hs, rs))


# (label, points, color, marker, hollow). Unified scheme: GR green, RP2 red /
# RP5 orange, IP purple, PPS brown, base gray. lr3 RP configs are HOLLOW (white
# face + colored edge) so lr1 vs lr3 read apart in the legend — the whole point
# being that lr3 does NOT help on hf100.
ARMS = [
    ("GR lccoh64 (deployed)",   gr_points(),               "#2ca02c", "^", False),
    ("RP2+anchor, lr1",         rp_points("rp2_lc64_lr1"),  "#d62728", "s", False),
    ("RP2+anchor, lr3",         rp_points("rp2_lc64_lr3"),  "#d62728", "s", True),
    ("RP5+anchor, lr1",         rp_points("rp5_lc64_lr1"),  "#ff7f0e", "D", False),
    ("RP5+anchor, lr3",         rp_points("rp5_lc64_lr3"),  "#ff7f0e", "D", True),
    ("IP: mand-tw",             ip_points("mand-tw"),       "#9467bd", "P", False),
    ("IP: mand-tw-norm",        ip_points("mand-tw-norm"),  "#9467bd", "X", False),
    ("PPS: L20 α2",             pps_points(),               "#8c564b", "v", False),
    ("base (no RL)",            base_point(),               "#999999", "o", True),
]


def main():
    from matplotlib.lines import Line2D
    plt.rcParams["font.size"] = 13
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    handles = []
    print(f"{'arm':22s} {'n':>2} {'hack':>14} {'retain':>14}")
    for name, pts, color, marker, hollow in ARMS:
        s = stats(pts)
        if s is None:
            print(f"{name:22s}  (no data yet)")
            continue
        hm, he, rm, re_, per = s
        print(f"{name:22s} {len(per):>2} {hm:.3f}±{he:.3f}   {rm:.3f}±{re_:.3f}")
        face = "white" if hollow else color
        for h, r in per:
            ax.scatter(h, r, marker=marker, s=30, alpha=0.25, zorder=2,
                       facecolors="none" if hollow else color, edgecolors=color)
        ax.errorbar(hm, rm, xerr=he, yerr=re_, color=color, marker=marker,
                    markersize=13, markerfacecolor=face,
                    markeredgecolor=color, markeredgewidth=1.8 if hollow else 0,
                    capsize=3, elinewidth=1.4, zorder=4)
        handles.append(Line2D([0], [0], marker=marker, color="w",
                              markerfacecolor=face, markeredgecolor=color,
                              markeredgewidth=1.8 if hollow else 0, markersize=12,
                              label=f"{name}  ({hm:.2f} / {rm:.2f})"))
    # upper-left (high hack / high retain) is the empty corner; lower-left would
    # cover the PPS mean at ~(0.39, 0.53).
    ax.legend(handles=handles, loc="upper left", fontsize=10, frameon=True,
              framealpha=0.92, title="arm  (hack / retain)", title_fontsize=10)

    ax.set_xlim(0.60, -0.03)
    ax.set_ylim(0.35, 0.92)
    ax.set_xlabel("ground-truth hack rate over all prompts  (better →)")
    ax.set_ylabel("retain: true_tested over all prompts  (better ↑)")
    ax.set_title("Countdown hf100 (100% hackable) — deployed hack vs retain (mean ± SEM)\n"
                 "only GR holds; single-policy interventions each fail")
    ax.grid(True, alpha=0.25)
    for d in ("/workspace/small-rl/figures_pareto/figs",
              "/workspace/small-rl/final_figures"):
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{d}/hf100_arms_scatter.{ext}", dpi=140, bbox_inches="tight")
    print("wrote figs/ + final_figures/ hf100_arms_scatter.png / .pdf")


if __name__ == "__main__":
    main()
