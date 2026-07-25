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
    import statistics as st
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(8.4, 7.4))

    # merge the two GRAFT globs into one 8-seed arm (countdown_figure1 style)
    merged = {}
    for label, pat, fs_scale, pr_scale, color, marker, hollow in ARMS:
        key = label.replace(" (ours, seeds 1-5)", " (ours)")
        ent = merged.setdefault(key, {"pats": [], "fs": fs_scale, "pr": pr_scale,
                                      "color": color, "marker": marker, "hollow": hollow})
        ent["pats"].append(pat)

    base_sm = json.load(open(BASE_FSEVAL))["scales"]["0.0"]
    base_hack = [v for k, v in base_sm.items() if k.split("/", 1)[0] == "hack_freq"][0]
    base_p = probe_pass1(BASE_PROBE_RUN, "0.0")
    merged["Base model"] = {"pts": [(base_hack, base_p)], "color": "#444444",
                            "marker": "o", "hollow": True}

    handles, all_h, all_p = [], [], []
    sem = lambda x: (st.stdev(x) / len(x) ** 0.5) if len(x) > 1 else 0.0
    print(f"{'arm':<26} {'n':>2} {'hack':>13} {'pass@1':>13}")
    for label, ent in merged.items():
        pts = ent.get("pts")
        if pts is None:
            pts = []
            for pat in ent["pats"]:
                for run, h in fseval_hack(pat, ent["fs"]).items():
                    pr = probe_pass1(run, ent["pr"])
                    if pr is not None:
                        pts.append((h, pr))
        if not pts:
            print(f"{label:<26} -- no data")
            continue
        hs, ps = zip(*pts)
        all_h += hs; all_p += ps
        color, marker, hollow = ent["color"], ent["marker"], ent["hollow"]
        for h, r in pts:
            ax.scatter(h, r, marker=marker, s=72, alpha=0.4, zorder=2,
                       facecolors="none" if hollow else color,
                       edgecolors="none" if not hollow else color)
        ax.errorbar(st.mean(hs), st.mean(ps), xerr=sem(hs), yerr=sem(ps),
                    color=color, marker=marker, markersize=21,
                    markerfacecolor="white" if hollow else color,
                    markeredgecolor=color if hollow else "white",
                    markeredgewidth=2.0 if hollow else 1.6,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    zorder=50 if label == "GRAFT (ours)" else 4,
                    clip_on=label != "GRAFT (ours)")
        handles.append(Line2D(
            [0], [0], marker=marker, color="w", linestyle="none",
            markerfacecolor="white" if hollow else color,
            markeredgecolor=color if hollow else "white",
            markeredgewidth=2.0 if hollow else 1.6, markersize=17, label=label))
        print(f"{label:<26} {len(pts):>2} {st.mean(hs):.3f}±{sem(hs):.3f} "
              f"{st.mean(ps):.3f}±{sem(ps):.3f}")

    ax.set_xlim(max(all_h) + 0.05, -0.03)
    ax.set_ylim(min(all_p) - 0.04, max(all_p) + 0.04)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Reward hack rate  (better →)", fontsize=25)
    ax.set_ylabel("Expr-only solve rate  (better →)", fontsize=25, labelpad=8)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, loc="lower right", fontsize=15, framealpha=0.92)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, "figs", f"countdown_expronly_scatter.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
