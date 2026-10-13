"""RP data-mixture figure for the canonical respmon family (analogue of the
paper draft's appendix ratio figure): per-env pareto panels showing the RP
main:extras mixture series {plain 1:0, 4:1, 1:1} plus the specially-labeled
0:1 arm ("training only on the unhackable distribution"), with do-nothing and
GR (deploy) as context.

All points: x = hack rate on hackable eval prompts, y = retain reward on
hackable prompts; both-mode for DN and every RP/unhackonly arm, retain_only
for GR. Mean ± sd over seeds. Eval is the unmodified hf=0.5 distribution for
every arm (the 0:1 arm trains at hack_frac 0.0 but evals via --eval_hack_frac
0.5).

Usage:
    .venv/bin/python figures_pareto/respmon_canon_rp_mix_figure.py
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMPOSITE = "output/respmon_canon_vercoh128_composite"
MIX_SWEEP = "output/respmon_canon_rp_mix_3seed"
UNHACK_SWEEP = "output/respmon_canon_unhackonly_3seed"

ENVS = ["addition_v2", "object_qa", "cities_qa", "persona_qa", "repeat", "sorting_copy"]
ENV_LABEL = {"addition_v2": "addition", "object_qa": "object (inv)", "cities_qa": "cities",
             "persona_qa": "persona", "repeat": "repeat", "sorting_copy": "sort"}

RP_COLOR = "#F59E0B"
SERIES = [  # (key, label, color, marker, eval mode)
    ("donothing",  "Do-nothing",                        "#6B7280", "o", "both"),
    ("rp_plain",   "RP 2.0 (no extras, 1:0)",           RP_COLOR,  "o", "both"),
    ("rp_mix4to1", "RP 2.0, main:unhackable 4:1",       RP_COLOR,  "p", "both"),
    ("rp_mix1to1", "RP 2.0, main:unhackable 1:1",       RP_COLOR,  "P", "both"),
    ("unhackonly", "Unhackable data only (0:1, no RP)", "#059669", "D", "both"),
    ("gr",         "GR (deploy)",                       "#2563EB", "o", "retain_only"),
]


def last_row(pe):
    rows = {}
    for line in open(pe):
        r = json.loads(line)
        rows[(r.get("step"), r.get("mode", ""))] = r
    last = max(k[0] for k in rows)
    return [v for k, v in rows.items() if k[0] == last][-1]


def classify(name):
    if "unhackonly" in name:
        return "unhackonly"
    if "donothing" in name:
        return "donothing"
    if "rp2" in name:
        if "mix4to1" in name:
            return "rp_mix4to1"
        if "mix1to1" in name:
            return "rp_mix1to1"
        return "rp_plain"
    if "gr_cls" in name or "invgr_cls" in name:
        return "gr"
    return None


def collect():
    pts = defaultdict(lambda: defaultdict(list))  # env -> series -> [(x, y)]
    dirs = (glob.glob(os.path.join(COMPOSITE, "*/"))
            + glob.glob(os.path.join(MIX_SWEEP, "*/"))
            + glob.glob(os.path.join(UNHACK_SWEEP, "*/"))
            + glob.glob("output/respmon_canon_rp4to1_seeds4to8/*/"))
    seen = set()
    for d in sorted(dirs):
        name = os.path.basename(d.rstrip("/"))
        if name in seen:  # composite symlinks duplicate sweep dirs
            continue
        seen.add(name)
        pe = os.path.join(d, "routing_eval.jsonl")
        if not os.path.exists(pe):
            continue
        env = next((e for e in ENVS if name.startswith(e)), None)
        series = classify(name)
        if env is None or series is None:
            continue
        mode = dict((s[0], s[4]) for s in SERIES)[series]
        r = last_row(pe)
        g = lambda p: next((v for k, v in r.items() if k.startswith(p)), None)
        x, y = g(f"{mode}/hack_freq_hackable/"), g(f"{mode}/retain_hackable/")
        if x is None or y is None:
            continue
        pts[env][series].append((x, y))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        COMPOSITE, "sweep_graphs", "canon_rp_mix_figure.png"))
    args = ap.parse_args()
    pts = collect()

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8), sharex=True, sharey=True)
    for ax, env in zip(axes.flat, ENVS):
        for key, label, color, marker, _mode in SERIES:
            vals = pts.get(env, {}).get(key, [])
            if not vals:
                continue
            xs, ys = zip(*vals)
            ax.errorbar(np.mean(xs), np.mean(ys), xerr=np.std(xs), yerr=np.std(ys),
                        fmt=marker, ms=8, color=color, ecolor=color,
                        elinewidth=1.1, capsize=3, mew=0.5, mec="white")
        ax.set_title(ENV_LABEL[env], fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, lw=0.4, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("hack rate (hackable prompts)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("retain reward (hackable)", fontsize=9)
    handles = [plt.Line2D([], [], marker=m, ls="", color=c, mec="white", mew=0.5,
                          ms=8, label=l) for _k, l, c, m, _md in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("RP data mixtures on the respmon canonical envs — "
                 "main:unhackable {1:0, 4:1, 1:1, 0:1} (mean ± sd over seeds)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    fig.savefig(args.out.replace(".png", ".pdf"))
    print("wrote", args.out)
    for env in ENVS:
        for key, label, *_ in SERIES:
            v = pts.get(env, {}).get(key, [])
            if v:
                xs, ys = zip(*v)
                print(f"  {env:12s} {key:11s} hack={np.mean(xs):.3f}±{np.std(xs):.3f} "
                      f"retain={np.mean(ys):.3f}±{np.std(ys):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
