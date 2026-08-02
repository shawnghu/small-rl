"""Two-panel countdown scatter: hf100 (left) vs hf50 (right), clean-env y.

Appendix figure (Jake 2026-08-02): y = the expr-only (clean-environment)
probe's pass@1 on both panels, the same metric as Figure 2's y-axis — the
scaffold is removed so hacking is structurally impossible. The hf50 probes
were run 2026-08-02 on Modal (tools/modal_expr_probe.py, 15 runs, n=256 k=1).
The base row is shared across panels: both variants train from the same frozen
SFT checkpoint and the probe prompts are identical.

Arms: the subset shared by both envs, seed-matched at s9/15/16 —
  No intervention / Reward penalty (rp2+LC-anchor lr1, the hf100 dev-selected
  config and its exact hf50 analog) / Inoculation prompting (mand-tw) /
  Preventative steering (L20 a2) / GRAFT (lccoh64 lr3, forget ablated) /
  base model (frozen SFT-primed checkpoint, evaluated once).
x = hack_freq_hackable (hackable-prompt hack rate; on hf100 this equals the
marginal rate since every prompt is hackable), inverted. Colors/markers match
countdown_figure1's scatter so the arms read the same across figures.

OUT is absolute (the shared checkout's output/), so the script runs unchanged
from a worktree.

Run: /workspace/small-rl/.venv/bin/python figures_pareto/countdown_hf100_hf50_retain.py
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/workspace/small-rl/output"


def val(d, p):
    hits = [v for k, v in d.items() if k.startswith(p) and "/" not in k[len(p):]]
    return hits[0] if hits else None


PROBES = "/workspace/small-rl/output/expr_only_probe_arms"


def probe_pass1(run_name, scale):
    f = os.path.join(PROBES, f"{run_name}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))["scales"].get(f"{float(scale):.1f}")
    return d["true_pass1"] if d else None


def pts(pattern, scale):
    out = []
    for f in sorted(glob.glob(pattern)):
        if "__" in os.path.basename(f):
            continue
        sm = json.load(open(f))["scales"]
        if scale not in sm:
            continue
        y = probe_pass1(os.path.basename(f)[:-5], scale)
        if y is None:
            print(f"  WARNING no probe for {os.path.basename(f)} @{scale} — dropped")
            continue
        out.append((val(sm[scale], "hack_freq_hackable/"), y))
    return out


# The frozen base is one model shared by both variants; its x (hackable-prompt
# hack rate) comes from each variant's own r0.0 fseval, its y from the single
# base probe.
BASE_PROBE = probe_pass1("cdhf100_gr_lccoh64_lr3_s16__r0.0", "0.0")


def base_pt(path):
    d = json.load(open(path))["scales"]["0.0"]
    return [(val(d, "hack_freq_hackable/"), BASE_PROBE)]


# (label, color, marker, hollow) — countdown_figure1's scatter styling.
STYLE = [
    ("No intervention",       "#e0905a", "X", False),
    ("Reward penalty",        "#d62728", "s", False),
    ("Inoculation prompting", "#a08070", "v", False),
    ("Preventative steering", "#8aa5a8", "h", False),
    ("GRAFT (ours)",          "#2ca02c", "o", False),
    ("Base model",            "#444444", "o", True),
]

PANELS = [
    ("Countdown-Code (100% hackable)", {
        "No intervention":       (f"{OUT}/countdown_code_rp2-0702-0026_fseval/"
                                  "countdown_code_hack_reward_penalty_amountmissing_s*.json", "1.0"),
        "Reward penalty":        (f"{OUT}/countdown_hf100_rp_lccoh64_fseval/cdhf100_rp2_lc64_lr1_s*.json", "1.0"),
        "Inoculation prompting": (f"{OUT}/countdown_hf100_ip_fseval/cdhf100_ip_mand-tw_s*.json", "1.0"),
        "Preventative steering": (f"{OUT}/countdown_hf100_pps_fseval/cdhf100_pps_L20_a2_s*.json", "1.0"),
        "GRAFT (ours)":          (f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_*.json", "0.0"),
        "Base model":            f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_gr_lccoh64_lr3_s9__r0.0.json",
    }),
    ("Countdown-Code (50% hackable)", {
        "No intervention":       (f"{OUT}/countdown_hf50_dn_fseval/cdhf50_dn_s*.json", "1.0"),
        "Reward penalty":        (f"{OUT}/countdown_hf50_rp_lccoh64_fseval/cdhf50_rp2_lc64_lr1_s*.json", "1.0"),
        "Inoculation prompting": (f"{OUT}/countdown_hf50_ip_fseval/cdhf50_ip_mand-tw_s*.json", "1.0"),
        "Preventative steering": (f"{OUT}/countdown_hf50_pps_fseval/cdhf50_pps_L20_a2_s*.json", "1.0"),
        "GRAFT (ours)":          (f"{OUT}/countdown_hf50_gr_lccoh_lr3_fseval/cdhf50_gr_lccoh64_lr3_s*.json", "0.0"),
        "Base model":            f"{OUT}/countdown_hf50_gr_nocoh_fseval/cdhf50_gr_nocoh_s9__r0.0.json",
    }),
]


def sem(xs):
    return st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


def main():
    plt.rcParams["font.size"] = 17
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), sharey=True)

    for ax, (title, arms) in zip(axes, PANELS):
        for label, color, marker, hollow in STYLE:
            src = arms[label]
            p = base_pt(src) if label == "Base model" else pts(*src)
            if not p:
                print(f"  [{title}] {label}: NO DATA"); continue
            hs = [q[0] for q in p]; rs = [q[1] for q in p]
            for h, r in p:
                ax.scatter(h, r, marker=marker, s=64, alpha=0.4, zorder=2,
                           facecolors="none" if hollow else color,
                           edgecolors=color if hollow else "none")
            ax.errorbar(st.mean(hs), st.mean(rs),
                        xerr=sem(hs), yerr=sem(rs), color=color, marker=marker,
                        markersize=17, markerfacecolor="white" if hollow else color,
                        markeredgecolor=color if hollow else "white",
                        markeredgewidth=1.8 if hollow else 1.4,
                        capsize=4, capthick=1.1, elinewidth=1.1,
                        zorder=50 if label == "GRAFT (ours)" else 4)
            print(f"  [{title}] {label:24s} n={len(p)} hack={st.mean(hs):.3f}±{sem(hs):.3f} "
                  f"retain={st.mean(rs):.3f}±{sem(rs):.3f}")
        ax.set_title(title, fontsize=19)
        ax.set_xlim(1.03, -0.03)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("Reward hack rate (hackable prompts)  (better →)", fontsize=17)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylim(0.0, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[0].set_ylabel("Clean environment solve rate  (better →)", fontsize=17)

    handles = [Line2D([0], [0], marker=m, linestyle="none",
                      markerfacecolor="white" if hollow else c,
                      markeredgecolor=c if hollow else "white",
                      markeredgewidth=1.8 if hollow else 1.4,
                      markersize=13, label=lab)
               for lab, c, m, hollow in STYLE]
    axes[1].legend(handles=handles, loc="lower left", fontsize=13, framealpha=0.92)
    fig.tight_layout()

    for d in (os.path.join(HERE, "figs"),):
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"countdown_hf100_hf50_retain.{ext}"),
                        dpi=140, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {d}/countdown_hf100_hf50_retain.png/.pdf")


if __name__ == "__main__":
    main()
