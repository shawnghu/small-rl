"""Countdown hf100 arms: expr-only capability (y) vs deployed hack rate (x).

y = expr-only pass@1 (tools/expr_only_probe.py, k=1, n=256, 2026-07-25): the
model is asked for ONLY the final expression (json {"expr": ...}), so hacking
is not expressible; scored in-process with the canonical ground-truth checker
on the SAME 256 eval problems as the fsevals. Answers the reviewer's
capability/hacking entanglement concern: the y-axis cannot be inflated by any
form of verifier tampering, and (unlike retain-on-hackable-prompts) is read in
a context where the hack affordance does not exist.

x = the existing fseval hack_freq (deployment config: GR at fs0.0, baselines
at fs1.0) — unchanged from countdown_figure1.

Arms, colors, markers mirror countdown_figure1.scatter_arms. GRAFT w/o routing
is absent (checkpoints lost; rerun pending). DN is 2 seeds (s15 weights lost).

Run: cd figures_pareto && ../.venv/bin/python countdown_expronly_scatter.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
PROBE = os.path.join(OUT, "expr_only_probe_arms")
NOCOH_PROBE = os.path.join(OUT, "expr_only_probe")  # k=4 run of the nocoh arm (not plotted)


def fseval_hack(pattern, scale):
    """{seed_tag: hack_freq} from fseval JSONs (skip derivative files)."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        if "__" in os.path.basename(f):
            continue
        d = json.load(open(f))
        sm = d["scales"][scale]
        hits = [v for k, v in sm.items() if k.split("/", 1)[0] == "hack_freq"]
        assert len(hits) == 1
        out[d["run_name"]] = float(hits[0])
    return out


def probe_pass1(run_name, scale):
    f = os.path.join(PROBE, f"{run_name}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d["scales"][scale]["true_pass1"]


# (label, fseval glob, fseval scale, probe scale, color, marker, hollow)
ARMS = [
    ("No intervention",
     f"{OUT}/countdown_code_rp2-0702-0026_fseval/countdown_code_hack_reward_penalty_amountmissing_s*.json",
     "1.0", "1.0", "#e0905a", "X", False),
    ("Reward penalty",
     f"{OUT}/countdown_hf100_rp_lccoh64_fseval/cdhf100_rp2_lc64_lr1_s*.json",
     "1.0", "1.0", "#d62728", "s", False),
    ("Inoculation prompting",
     f"{OUT}/countdown_hf100_ip_fseval/cdhf100_ip_mand-tw_s*.json",
     "1.0", "1.0", "#a08070", "v", False),
    ("Preventative steering",
     f"{OUT}/countdown_hf100_pps_fseval/cdhf100_pps_L20_a2_s*.json",
     "1.0", "1.0", "#8aa5a8", "h", False),
    ("Anchor environment only",
     f"{OUT}/countdown_hf100_lconly_fseval/lconly_full_s*.json",
     "1.0", "1.0", "#9467bd", "P", False),
    ("GRAFT (ours)",
     f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_*.json",
     "0.0", "0.0", "#2ca02c", "o", False),
    ("GRAFT (ours, seeds 1-5)",
     f"{OUT}/countdown_hf100_gr_lccoh64_lr3_seeds5_fseval/cdhf100_*.json",
     "0.0", "0.0", "#2ca02c", "o", False),
]

BASE_PROBE_RUN = "cdhf100_gr_lccoh64_lr3_s16__r0.0"
BASE_FSEVAL = (f"{OUT}/countdown_hf100_gr_lccoh64_lr3_fseval/"
               "cdhf100_gr_lccoh64_lr3_s9__r0.0.json")


def main():
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    plt.rcParams.update({"font.size": 14})

    seen_labels = set()
    print(f"{'arm':<26} {'run':<48} {'hack':>6} {'pass@1':>7}")
    for label, pat, fs_scale, pr_scale, color, marker, hollow in ARMS:
        hacks = fseval_hack(pat, fs_scale)
        pts = []
        for run, h in hacks.items():
            p = probe_pass1(run, pr_scale)
            if p is None:
                print(f"{label:<26} {run:<48} {h*100:5.1f}%   (no probe)")
                continue
            pts.append((h, p))
            print(f"{label:<26} {run:<48} {h*100:5.1f}% {p*100:6.1f}%")
        if not pts:
            continue
        hs, ps = zip(*pts)
        show_label = label.replace(", seeds 1-5", "") if label not in seen_labels else None
        display = label.replace(" (ours, seeds 1-5)", " (ours)")
        kw = dict(color=color, marker=marker, s=130, zorder=6,
                  facecolors="none" if hollow else color,
                  edgecolors=color, linewidths=1.6)
        if display not in seen_labels:
            kw["label"] = display
            seen_labels.add(display)
        ax.scatter([h * 100 for h in hs], [p * 100 for p in ps], alpha=0.85, **kw)

    # base model
    bh = fseval_hack(BASE_FSEVAL.replace("__r0.0", "__MISS"), "0.0")  # none — handled below
    base_sm = json.load(open(BASE_FSEVAL))["scales"]["0.0"]
    base_hack = [v for k, v in base_sm.items() if k.split("/", 1)[0] == "hack_freq"][0]
    base_p = probe_pass1(BASE_PROBE_RUN, "0.0")
    ax.scatter([base_hack * 100], [base_p * 100], color="#444444", marker="o",
               s=150, facecolors="none", edgecolors="#444444", linewidths=1.8,
               label="Base model", zorder=6)
    print(f"{'Base model':<26} {BASE_PROBE_RUN:<48} {base_hack*100:5.1f}% {base_p*100:6.1f}%")

    ax.set_xlabel("Hack rate, deployed configuration (%)", fontsize=15)
    ax.set_ylabel("Expression-only solve rate, pass@1 (%)", fontsize=15)
    ax.set_xlim(-3, 100)
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, "figs", f"countdown_expronly_scatter.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
