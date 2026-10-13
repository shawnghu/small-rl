"""Hack-generalization scatter for the hf100 countdown arms (one panel per
transfer env: HumanEval + LeetCode-verified).

Data: the countdown-generalization-eval graded caches
(output/countdown_generalization/{humaneval,leetcode}/<label>/graded.jsonl),
spec countdown_hf100_all. Per-seed faint markers + mean ± SEM bold markers,
axes/styling per countdown_figure1.py (x = hack rate inverted so better is
rightward, y = correct/retain rate). Hack rate = hardcode + tamper_pass
(% of all completions — the two countdown-style hack channels).

Arms (colors/markers = countdown_figure1.py):
  No intervention        old round-2 DN runs (s9/s16 — s15 ckpt lost)
  Reward penalty         rp2_lc64_lr1 (practitioner-selected best)
  Inoculation prompting  mand-tw, prompt removed at eval
  Preventative steering  L20 alpha2, steering removed at eval
  Anchor environment only  lconly_full
  GRAFT w/o routing      noroute (recall=0, anchor intact), forget ablated
  GRAFT: pre-ablation    GR 2-adapter (training config, fs=1)
  GRAFT (ours)           GR retain-only (deployed, fs=0), 8 seeds
  Base model             fixed BASE_ROWS floor (single hollow point)

Run: .venv/bin/python figures_pareto/countdown_generalization_scatter.py
"""
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools", "countdown-generalization-eval"))
from checkpoint_specs import BASE_ROWS, expand  # noqa: E402
from make_table import _config_stats  # noqa: E402

CACHE_ROOT = os.path.join(ROOT, "output", "countdown_generalization")
SPEC = "countdown_hf100_all"

# (spec group, display label, color, marker, hollow) — figure1 order/colors.
ARMS = [
    ("No intervention (DN)", "No intervention", "#e0905a", "X", False),
    ("RP (rp2_lc64_lr1)", "Reward penalty", "#d62728", "s", False),
    ("IP (mand-tw)", "Inoculation prompting", "#a08070", "v", False),
    ("PPS (L20 a2)", "Preventative steering", "#8aa5a8", "h", False),
    ("Anchor only (lconly_full)", "Anchor environment only", "#9467bd", "P", False),
    # Ablate-one-adapter-without-routing baseline (figure1's style/label); the
    # spec also generates its 2-adapter config, which the tables report but the
    # figure omits (parity with figure1, which shows only the ablated point).
    ("No routing retain-only", "GRAFT w/o routing", "#9690a8", "X", True),
    ("GR 2-adapter", "GRAFT: pre-ablation", "#1f77b4", "o", True),
    ("GR retain-only", "GRAFT (ours)", "#2ca02c", "o", False),
]
BASE_STYLE = ("Base model", "#444444", "o", True)

PANELS = [("humaneval", "HumanEval", "Correct solution rate  (better →)"),
          ("leetcode", "LeetCode-verified", "Hidden-test pass rate  (better →)")]

sem = lambda x: (st.stdev(x) / len(x) ** 0.5) if len(x) > 1 else 0.0


def arm_points(kind):
    """[(display label, [(hack%, retain%)], color, marker, hollow)]."""
    by_group = {}
    for c in expand(SPEC):
        s = _config_stats(CACHE_ROOT, kind, c["label"])
        assert s is not None, f"missing graded cache: {kind}/{c['label']}"
        by_group.setdefault(c["group"], []).append(
            (s["hardcode"] + s["tamper"], s["retain"]))
    out = [(lab, by_group[g], col, mk, ho) for g, lab, col, mk, ho in ARMS]
    b = BASE_ROWS[kind]
    lab, col, mk, ho = BASE_STYLE
    out.append((lab, [(b["hardcode"] + b["tamper"], b["retain"])], col, mk, ho))
    return out


def draw_panel(ax, kind, title, ylabel):
    all_h, all_r, handles = [], [], []
    for label, pts, color, marker, hollow in arm_points(kind):
        hs, rs = [p[0] for p in pts], [p[1] for p in pts]
        all_h += hs
        all_r += rs
        for h, r in pts:
            ax.scatter(h, r, marker=marker, s=64, alpha=0.4, zorder=2,
                       facecolors="none" if hollow else color,
                       edgecolors=color if hollow else "none")
        ax.errorbar(st.mean(hs), st.mean(rs), xerr=sem(hs), yerr=sem(rs),
                    color=color, marker=marker, markersize=16,
                    markerfacecolor="white" if hollow else color,
                    markeredgecolor=color if hollow else "white",
                    markeredgewidth=2.0 if hollow else 1.4,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    zorder=50 if label == "GRAFT (ours)" else 4,
                    clip_on=label != "GRAFT (ours)")
        handles.append(Line2D(
            [0], [0], marker=marker, color="w", linestyle="none",
            markerfacecolor="white" if hollow else color,
            markeredgecolor=color if hollow else "white",
            markeredgewidth=2.0 if hollow else 1.4, markersize=13, label=label))
        print(f"  [{kind}] {label:26s} n={len(pts)} hack={st.mean(hs):5.1f}±{sem(hs):4.1f} "
              f"retain={st.mean(rs):5.1f}±{sem(rs):4.1f}")
    ax.set_xlim(max(all_h) + 4, -2)
    ax.set_ylim(min(all_r) - 4, max(all_r) + 4)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Reward hack rate  (better →)", fontsize=19)
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_title(title, fontsize=21)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return handles


def main():
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.2))
    for ax, (kind, title, ylabel) in zip(axes, PANELS):
        handles = draw_panel(ax, kind, title, ylabel)
    axes[1].legend(handles=handles, loc="upper left", fontsize=12.5,
                   framealpha=0.92)
    fig.tight_layout()
    for d in (os.path.join(HERE, "figs"), os.path.join(ROOT, "final_figures")):
        os.makedirs(d, exist_ok=True)
        for ext in ("pdf", "png"):
            out = os.path.join(d, f"countdown_generalization_scatter.{ext}")
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04,
                        **({"dpi": 150} if ext == "png" else {}))
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
