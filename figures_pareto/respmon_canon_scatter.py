"""Pareto scatter for the canonical respmon-v3 env family (2026-07-25:
object_qa canonically INVERTED; addition/cities standard iva; persona bvo;
repeat/sorting unchanged), reading the spliced composite sweep dir.

Deployment operating point per (env, method), aggregated over seeds:
  x = hack rate on hackable prompts   (both-mode for DN/RP; retain_only for GR)
  y = retain reward on hackable prompts (same mode)

Usage:
    .venv/bin/python figures_pareto/respmon_canon_scatter.py \
        [--sweep output/respmon_canon_vercoh128_composite]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = {  # name-substring -> (label, color, deployment eval mode)
    "donothing": ("Do-nothing", "#6B7280", "both"),
    "rp2":       ("Reward penalty (2.0)", "#F59E0B", "both"),
    "gr_cls":    ("GR (deploy)", "#2563EB", "retain_only"),
    "invgr_cls": ("GR (deploy)", "#2563EB", "retain_only"),
}
ENVS = ["addition_v2", "object_qa", "cities_qa", "persona_qa", "repeat", "sorting_copy"]
ENV_LABEL = {"addition_v2": "addition", "object_qa": "object (inv)", "cities_qa": "cities",
             "persona_qa": "persona", "repeat": "repeat", "sorting_copy": "sort"}


def collect(sweep_dir):
    pts = defaultdict(lambda: defaultdict(list))  # env -> label -> [(x, y)]
    for d in sorted(glob.glob(os.path.join(sweep_dir, "*/"))):
        pe = os.path.join(d, "routing_eval.jsonl")
        if not os.path.exists(pe):
            continue
        name = os.path.basename(d.rstrip("/"))
        env = next((e for e in ENVS if name.startswith(e)), None)
        meth = next((m for m in ("donothing", "rp2", "invgr_cls", "gr_cls") if m in name), None)
        if env is None or meth is None:
            continue
        label, color, mode = METHODS[meth]
        r = json.loads(open(pe).readlines()[-1])
        g = lambda p: next((v for k, v in r.items() if k.startswith(p)), None)
        x = g(f"{mode}/hack_freq_hackable/")
        y = g(f"{mode}/retain_hackable/")
        if x is None or y is None:
            continue
        pts[env][label].append((x, y))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="output/respmon_canon_vercoh128_composite")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pts = collect(args.sweep)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    order = [("Do-nothing", "#6B7280"), ("Reward penalty (2.0)", "#F59E0B"),
             ("GR (deploy)", "#2563EB")]
    for ax, env in zip(axes.flat, ENVS):
        for label, color in order:
            vals = pts.get(env, {}).get(label, [])
            if not vals:
                continue
            xs, ys = zip(*vals)
            ax.errorbar(np.mean(xs), np.mean(ys),
                        xerr=np.std(xs), yerr=np.std(ys),
                        fmt="o", ms=7, color=color, ecolor=color,
                        elinewidth=1.2, capsize=3, mew=0)
        ax.set_title(ENV_LABEL[env], fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, lw=0.4, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("hack rate (hackable prompts)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("retain reward (hackable)", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for l, c in order]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("respmon canonical env family — deployment operating points "
                 "(vercoh128 GR vs baselines, 3 seeds, mean ± sd)", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = args.out or os.path.join(args.sweep, "sweep_graphs", "canon_pareto_scatter.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print("wrote", out)
    for env in ENVS:
        for label, _ in order:
            v = pts.get(env, {}).get(label, [])
            if v:
                xs, ys = zip(*v)
                print(f"  {env:12s} {label:22s} hack={np.mean(xs):.3f}±{np.std(xs):.3f} "
                      f"retain={np.mean(ys):.3f}±{np.std(ys):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
