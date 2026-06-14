"""Scatter trajectories for the canonical-steps 5-seed sweep.

Single-panel: one trajectory per (env, seed) through all 6 forget_scales.
Endpoints only: hollow circle at f=0 (retain_only), filled at f=1.
Reference per env: GR canonical retain_only anchor from
figures_pareto/aggregated_cache.json (the same points used in
proto_pareto_7envs_gr_rp_v2.pdf), drawn as a star in the env's color.

X-axis is inverted so low-hack lives on the right (matching the v2 paper
figure convention: better = up-and-right).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/workspace/small-rl")
# Default; override with argv[1] = eval dir relative to repo root and optional
# argv[2] = title tag, e.g.
#   tools/plot_canonical_5seed_forget_scale.py \
#     output/gr_forget_scale_eval/canonical_radam_1k_samples "RoutedAdam bw2, 3 seeds"
SRC = REPO / "output/gr_forget_scale_eval/canonical_5seed_1k_samples/results.jsonl"
# Reference: classic + interlaced-coherence GR canonical anchors (cspr=32, 5 seeds,
# canonical max_steps — 2000 for the long-train envs, 1000 for repeat/topic). These
# are the GR points backing proto_pareto_7envs_gr_rp_v2.pdf.
REFERENCE_CACHE = REPO / "figures_pareto/aggregated_cache.json"
OUT_BASE = REPO / "output/gr_forget_scale_eval/canonical_5seed_1k_samples/scatter_trajectories"
TITLE_TAG = "classic routing, no coherence, canonical max_steps, 5 seeds"

ENV_COLORS = {
    "persona_qa":     "#1f77b4",
    "sorting_copy":   "#d62728",
    "repeat_extra":   "#2ca02c",
    "cities_qa":      "#9467bd",
    "object_qa":      "#ff7f0e",
    "addition_v2":    "#8c564b",
    "topic_contains": "#e377c2",
}


def _load_reference_points():
    """GR canonical retain_only (= forget_scale=0) anchors per env, 5 seeds,
    canonical max_steps. Cache format per env: 'gr' -> [retain_mean,
    retain_std, hack_mean, hack_std, n_seeds]."""
    if not REFERENCE_CACHE.exists():
        return {}
    with REFERENCE_CACHE.open() as f:
        cache = json.load(f)
    out = {}
    for env, e in cache.items():
        gr = e.get("gr")
        if not gr:
            continue
        retain_m, _, hack_m, _, _ = gr
        out[env] = (retain_m, hack_m)
    return out


def main():
    global SRC, OUT_BASE, TITLE_TAG
    if len(sys.argv) > 1:
        eval_dir = REPO / sys.argv[1]
        SRC = eval_dir / "results.jsonl"
        OUT_BASE = eval_dir / "scatter_trajectories"
    if len(sys.argv) > 2:
        TITLE_TAG = sys.argv[2]
    assert SRC.exists(), f"missing {SRC}"

    rows = []
    with SRC.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["env"], r["seed"])].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r["forget_scale"])

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.plot([0, 1], [0, 1], ls=":", color="0.85", lw=0.8, zorder=1)

    for (env, seed), traj in sorted(groups.items()):
        xs = [r["hack_overall"] for r in traj]
        ys = [r["retain"] for r in traj]
        color = ENV_COLORS.get(env, "0.4")
        if any(v is None for v in xs + ys):
            continue
        # Line through all forget_scales; markers only at the endpoints
        # (hollow at f=0, filled at f=1) — no intermediate circles.
        ax.plot(xs, ys, "-", color=color, alpha=0.45, lw=1.1, zorder=2)
        ax.scatter([xs[0]], [ys[0]], s=55, marker="o",
                   facecolors="white", edgecolors=color, linewidths=1.4,
                   zorder=3, label=f"{env} s{seed}")
        ax.scatter([xs[-1]], [ys[-1]], s=55, marker="o",
                   color=color, zorder=3)

    ref = _load_reference_points()
    if ref:
        envs_present = sorted({env for env, _ in groups})
        for env in envs_present:
            if env not in ref:
                continue
            retain_m, hack_m = ref[env]
            color = ENV_COLORS.get(env, "0.4")
            ax.scatter([hack_m], [retain_m],
                       s=200, marker="*", color=color, zorder=6)

    ax.set_xlabel("Hack rate (overall hack_freq)")
    ax.set_ylabel("Retain reward")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.invert_xaxis()
    ax.grid(True, color="0.94", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Forget-scale trajectories ({TITLE_TAG})\n"
        "n_eval=1000; ○ = f=0 (retain_only), ● = f=1; "
        "stars = GR canonical retain_only (classic+coh, 5 seeds, ref)"
    )

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup = []
    for h, l in zip(handles, labels):
        env_label = l.split(" ")[0]
        if env_label not in seen:
            seen.add(env_label)
            dedup.append(plt.Line2D([], [], marker="o", linestyle="-",
                                    color=ENV_COLORS.get(env_label, "0.4"),
                                    markersize=10, label=env_label))
    ax.legend(handles=dedup, loc="best", fontsize=10)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_BASE.with_suffix(f".{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
