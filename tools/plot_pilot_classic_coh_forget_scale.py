"""Scatter plot for the classic + coherence pilot."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/workspace/small-rl")
SRC = REPO / "output/gr_forget_scale_eval/pilot_classic_coh/results.jsonl"
BASE_SRC = REPO / "output/gr_forget_scale_eval/pilot/base_results.jsonl"
# Stars come from this same file's own f=0 mean (so the star sits on the trajectory's
# f=0 endpoint — a useful sanity marker that the canonical reference IS this regime).
REFERENCE_SRC = REPO / "output/gr_forget_scale_eval/pilot_classic_coh/results.jsonl"
OUT_BASE = REPO / "output/gr_forget_scale_eval/pilot_classic_coh/scatter_trajectories"


def _load_reference_points():
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
    "persona_qa":   "#1f77b4",
    "sorting_copy": "#d62728",
    "repeat_extra": "#2ca02c",
    "cities_qa":    "#9467bd",
    "object_qa":    "#ff7f0e",
    "addition_v2":  "#8c564b",
    "topic_contains":"#e377c2",
}


def main():
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
        scales = [r["forget_scale"] for r in traj]
        color = ENV_COLORS.get(env, "0.4")
        if any(v is None for v in xs + ys):
            print(f"[skip] {env} s{seed}: null value(s)")
            continue
        ax.plot(xs, ys, "-", color=color, alpha=0.45, lw=1.5, zorder=2)
        sizes = [40 + 220 * s for s in scales]
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.8,
                   edgecolors="white", linewidths=1.0, zorder=3,
                   label=f"{env} s{seed}")

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
    ax.set_title("Forget-scale trajectories (classic routing + coherence, 1000 steps)\n"
                 "Marker size grows with forget_scale")

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
