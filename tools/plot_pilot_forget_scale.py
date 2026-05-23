"""Scatter plot of pilot forget-scale eval.

X = hack rate (overall hack_freq)
Y = retain reward
Each run = one trajectory connecting its 6 (hack, retain) points as
forget_scale ramps 0 → 1.

Reads output/gr_forget_scale_eval/pilot/results.jsonl (one row per
(run, scale)) and writes output/gr_forget_scale_eval/pilot/scatter_trajectories.{pdf,png}.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/small-rl")
SRC = REPO / "output/gr_forget_scale_eval/pilot/results.jsonl"
BASE_SRC = REPO / "output/gr_forget_scale_eval/pilot/base_results.jsonl"
# Reference star source: classic+coherence pilot's f=0 (retain_only) mean across
# its 2 seeds per env. Apples-to-apples comparison at 1000 training steps,
# rather than the canonical 5-seed/2000-step aggregated_cache used previously.
REFERENCE_SRC = REPO / "output/gr_forget_scale_eval/pilot_classic_coh/results.jsonl"
OUT_BASE = REPO / "output/gr_forget_scale_eval/pilot/scatter_trajectories"


def _load_reference_points():
    """Returns dict env -> (retain_mean, hack_mean) at forget_scale=0."""
    if not REFERENCE_SRC.exists():
        return {}
    from collections import defaultdict
    rows = defaultdict(list)
    with REFERENCE_SRC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("forget_scale") != 0.0:
                continue
            if r.get("retain") is None or r.get("hack_overall") is None:
                continue
            rows[r["env"]].append((r["retain"], r["hack_overall"]))
    return {env: (sum(x[0] for x in vs) / len(vs),
                  sum(x[1] for x in vs) / len(vs))
            for env, vs in rows.items()}

ENV_COLORS = {
    "persona_qa":   "#1f77b4",  # blue
    "sorting_copy": "#d62728",  # red
    "repeat_extra": "#2ca02c",  # green
    "cities_qa":    "#9467bd",  # purple
    "object_qa":    "#ff7f0e",  # orange
    "addition_v2":  "#8c564b",  # brown
    "topic_contains":"#e377c2", # pink
}


def main():
    assert SRC.exists(), f"missing {SRC}"

    # Group rows by (env, seed) and order by forget_scale.
    rows = []
    with SRC.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["env"], r["seed"])].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r["forget_scale"])

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # Plot each run's trajectory.
    for (env, seed), traj in sorted(groups.items()):
        xs = [r["hack_overall"] for r in traj]
        ys = [r["retain"] for r in traj]
        scales = [r["forget_scale"] for r in traj]
        color = ENV_COLORS.get(env, "0.4")

        # Skip if any value missing.
        if any(v is None for v in xs + ys):
            print(f"[skip] {env} s{seed}: null value(s) in trajectory")
            continue

        # Line connecting the trajectory.
        ax.plot(xs, ys, "-", color=color, alpha=0.45, lw=1.5, zorder=2)

        # Points sized/colored by forget_scale (small at f=0, big at f=1).
        sizes = [40 + 220 * s for s in scales]
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.8,
                   edgecolors="white", linewidths=1.0, zorder=3,
                   label=f"{env} s{seed}")


    # Base-model reference points (open circles, one per env).
    if BASE_SRC.exists():
        with BASE_SRC.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                env = r["env"]
                color = ENV_COLORS.get(env, "0.4")
                ax.scatter([r["hack_overall"]], [r["retain"]],
                           s=180, facecolors="none", edgecolors=color,
                           linewidths=2.0, zorder=4)

    # Canonical GR (ours) reference points — published retain_only mode, 5-seed
    # mean from the canonical anchor runs (classic routing + coherence, 2000
    # steps). Drawn as filled stars in each env's color.
    ref = _load_reference_points()
    if ref:
        envs_present = sorted({env for env, _ in groups})
        for env in envs_present:
            if env not in ref:
                continue
            retain_m, hack_m = ref[env]
            color = ENV_COLORS.get(env, "0.4")
            ax.scatter([hack_m], [retain_m],
                       s=380, marker="*", facecolors=color, edgecolors="black",
                       linewidths=1.2, zorder=6)

    ax.set_xlabel("Hack rate (overall hack_freq)")
    ax.set_ylabel("Retain reward")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Forget-scale trajectories (pilot: exclusive routing, no coherence, 1000 steps)\n"
                 "Marker size grows with forget_scale (small=0, large=1)")
    # De-duplicate legend (one entry per env, not per seed)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup = []
    for h, l in zip(handles, labels):
        env_label = l.split(" ")[0]
        if env_label not in seen:
            seen.add(env_label)
            # Replace seed-specific label with env label for cleaner legend.
            h2 = plt.Line2D([], [], marker="o", linestyle="-",
                            color=ENV_COLORS.get(env_label, "0.4"),
                            markersize=10, label=env_label)
            dedup.append(h2)
    ax.legend(handles=dedup, loc="best", fontsize=10)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_BASE.with_suffix(f".{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
