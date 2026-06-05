"""Optimal forget-scale operating point per (variant, env, seed).

Score = retain - hack_overall. For each (variant, env, seed), pick the
forget_scale that maximizes the score and plot that point.

  color  -> variant (4)
  marker -> env (up to 7)
  star   -> reference: classic+coh at forget_scale=0, mean over seeds, per env

Output: output/gr_forget_scale_eval/scatter_optimal_per_variant.{pdf,png}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/workspace/small-rl")
EVAL_DIR = REPO / "output/gr_forget_scale_eval"
OUT_BASE = EVAL_DIR / "scatter_optimal_per_variant"

VARIANTS = [
    ("pilot",             "excl, no-coh",    "#d62728"),
    ("pilot_classic",     "classic, no-coh", "#1f77b4"),
    ("pilot_classic_coh", "classic + coh",   "#2ca02c"),
    ("pilot_excl_coh",    "excl + coh",      "#9467bd"),
]

ENV_MARKERS = {
    "persona_qa":     "o",
    "sorting_copy":   "s",
    "repeat_extra":   "^",
    "cities_qa":      "D",
    "object_qa":      "v",
    "addition_v2":    "P",
    "topic_contains": "X",
}

REFERENCE_VARIANT = "pilot_classic_coh"
REFERENCE_SCALE = 0.0


def _load(variant_dir: str):
    path = EVAL_DIR / variant_dir / "results.jsonl"
    assert path.exists(), f"missing {path}"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _optimal_per_group(rows):
    """Returns {(env, seed): row} where row maximizes retain - hack_overall."""
    by_group = defaultdict(list)
    for r in rows:
        if r.get("retain") is None or r.get("hack_overall") is None:
            continue
        by_group[(r["env"], r["seed"])].append(r)
    out = {}
    for k, lst in by_group.items():
        out[k] = max(lst, key=lambda r: r["retain"] - r["hack_overall"])
    return out


def _reference_means():
    """classic_coh f=0 per env, averaged over seeds."""
    rows = _load(REFERENCE_VARIANT)
    by_env = defaultdict(list)
    for r in rows:
        if r["forget_scale"] != REFERENCE_SCALE:
            continue
        if r.get("retain") is None or r.get("hack_overall") is None:
            continue
        by_env[r["env"]].append((r["hack_overall"], r["retain"]))
    return {env: (sum(x[0] for x in vs) / len(vs),
                  sum(x[1] for x in vs) / len(vs))
            for env, vs in by_env.items()}


def main():
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.plot([0, 1], [0, 1], ls=":", color="0.85", lw=0.8, zorder=1)

    # iso-score guides: retain - hack = const
    for c in (0.2, 0.4, 0.6, 0.8):
        ax.plot([0, 1 - c], [c, 1.0], ls="--", color="0.92", lw=0.6, zorder=1)

    envs_present = set()
    # variant -> env -> [scores per seed]; used for the variant-mean point.
    optimal_pts = {vdir: defaultdict(list) for vdir, _, _ in VARIANTS}
    for vdir, vlabel, vcolor in VARIANTS:
        rows = _load(vdir)
        opt = _optimal_per_group(rows)
        for (env, seed), r in sorted(opt.items()):
            envs_present.add(env)
            marker = ENV_MARKERS.get(env, "o")
            ax.scatter([r["hack_overall"]], [r["retain"]],
                       s=70, marker=marker, color=vcolor, alpha=0.85,
                       edgecolors="white", linewidths=0.7, zorder=4)
            optimal_pts[vdir][env].append((r["hack_overall"], r["retain"]))

    # Per-variant mean across envs (only envs present for ALL variants),
    # averaging seeds first so each env contributes equally regardless of
    # seed count.
    shared_envs = set.intersection(*(set(optimal_pts[v].keys())
                                     for v, _, _ in VARIANTS))
    print(f"\nvariant means over shared envs ({sorted(shared_envs)}):")
    for vdir, vlabel, vcolor in VARIANTS:
        env_means = []
        for env in shared_envs:
            seed_pts = optimal_pts[vdir][env]
            h = sum(p[0] for p in seed_pts) / len(seed_pts)
            r = sum(p[1] for p in seed_pts) / len(seed_pts)
            env_means.append((h, r))
        mh = sum(p[0] for p in env_means) / len(env_means)
        mr = sum(p[1] for p in env_means) / len(env_means)
        print(f"  {vlabel:18s}  hack={mh:.3f}  retain={mr:.3f}  score(r-h)={mr-mh:+.3f}")
        ax.scatter([mh], [mr], s=180, marker="*", color=vcolor,
                   zorder=7)

    # Gold reference mean over the same shared envs (per-env mean over seeds,
    # then average across envs).
    ref = _reference_means()
    ref_env_means = [ref[env] for env in shared_envs if env in ref]
    if ref_env_means:
        gh = sum(p[0] for p in ref_env_means) / len(ref_env_means)
        gr = sum(p[1] for p in ref_env_means) / len(ref_env_means)
        print(f"  {'gold (ref) f=0':18s}  hack={gh:.3f}  retain={gr:.3f}  score(r-h)={gr-gh:+.3f}")
        ax.scatter([gh], [gr], s=180, marker="*", color="#d4af37", zorder=8)

    # reference: classic_coh f=0 mean per env, gold + env-specific marker
    ref = _reference_means()
    ref_color = "#d4af37"  # gold
    for env in sorted(envs_present):
        if env not in ref:
            continue
        hack_m, retain_m = ref[env]
        marker = ENV_MARKERS.get(env, "o")
        ax.scatter([hack_m], [retain_m],
                   s=70, marker=marker, color=ref_color,
                   edgecolors="white", linewidths=0.7, zorder=6)

    ax.set_xlabel("Hack rate (overall hack_freq)")
    ax.set_ylabel("Retain reward")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color="0.94", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "Optimal forget-scale operating point per (variant, env, seed)\n"
        "score = retain - hack;  ★ = variant mean (envs shared by all variants, seeds avg'd first);  "
        "gold = classic+coh at forget_scale=0 (per-env mean)"
    )

    variant_handles = [
        plt.Line2D([], [], marker="o", linestyle="None",
                   color=c, markersize=10, label=lbl)
        for _, lbl, c in VARIANTS
    ]
    mean_handle = plt.Line2D([], [], marker="*", linestyle="None",
                             color="0.5", markersize=12, label="variant mean")
    variant_handles.append(mean_handle)
    env_handles = [
        plt.Line2D([], [], marker=ENV_MARKERS[env], linestyle="None",
                   color="0.3", markersize=10, label=env)
        for env in sorted(envs_present)
    ]
    ref_handle = plt.Line2D([], [], marker="o", linestyle="None",
                            color=ref_color, markersize=8,
                            label=f"{REFERENCE_VARIANT} f=0 (ref)")
    ref_mean_handle = plt.Line2D([], [], marker="*", linestyle="None",
                                 color=ref_color, markersize=12,
                                 label="ref mean")

    leg1 = ax.legend(handles=variant_handles, title="variant",
                     loc="lower right", fontsize=9, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=env_handles + [ref_handle, ref_mean_handle], title="env / ref",
              loc="upper right", fontsize=9, title_fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_BASE.with_suffix(f".{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
