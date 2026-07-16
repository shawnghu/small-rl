"""v2-suite GR pilot: hack vs retain scatter over the 7 re-keyed toy envs.

One axes, house style (proto_scatter_hack_retain_7envs): x = overall
hack_freq (reversed, right = better), y = retain. Two classes from
output/gr_v2suite_fseval/*.json:

  deployed (fs 0.0)      purple filled
  pre-ablation (fs 1.0)  purple hollow

Per-env: faint per-seed points, medium env-mean markers, a light arrow
fs1.0 -> fs0.0 showing what ablation does, and an env label — the pilot's
story is env heterogeneity (retain-collapse vs blind-spot leak vs hack
extinction), so env identity stays legible. Big markers = class means over
the 7 env means with 95% t-CI, house convention.

Run: .venv/bin/python figures_pareto/v2suite_scatter_7envs.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FSDIR = os.path.join(HERE, "..", "output", "gr_v2suite_fseval")
HACK, RET = "hack_freq", "retain"
PURPLE = "#9467bd"

plt.rcParams.update({"font.size": 20})

ENV_LABEL = {
    "addition_v2": "addition",
    "cities_qa_v2": "cities",
    "object_qa_v2": "object",
    "persona_qa": "persona",
    "repeat_v2": "repeat",
    "sorting_v2": "sorting",
    "topic": "topic",
}
# Label anchor offsets (axes-fraction-ish data units), hand-placed per env to
# keep labels off the markers/arrows.
LABEL_DXY = {
    "addition_v2": (0.03, 0.04),
    "cities_qa_v2": (0.03, -0.07),
    "object_qa_v2": (0.03, 0.04),
    "persona_qa": (0.03, -0.07),
    "repeat_v2": (0.03, -0.07),
    "sorting_v2": (0.03, 0.04),
    "topic": (0.03, 0.04),
}


def pick(sm, slug):
    for k, v in sm.items():
        if k.split("/", 1)[0] == slug:
            return float(v)
    return np.nan


def load():
    """(env -> scale -> list[(hack, retain)] per seed)"""
    byenv = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(FSDIR, "*.json"))):
        rec = json.load(open(f))
        m = re.match(r"(.+)_v2gr_s(\d+)$", rec["run_name"])
        assert m, rec["run_name"]
        env = m.group(1)
        for scale in ("0.0", "1.0"):
            sm = rec["scales"][scale]
            byenv[env][scale].append((pick(sm, HACK), pick(sm, RET)))
    assert len(byenv) == 7, sorted(byenv)
    return byenv


def main():
    byenv = load()
    fig, ax = plt.subplots(figsize=(9.5, 9.0))

    env_means = {"0.0": [], "1.0": []}
    for env in sorted(byenv):
        means = {}
        for scale, hollow in (("1.0", True), ("0.0", False)):
            arr = np.array(byenv[env][scale])
            ax.scatter(arr[:, 0], arr[:, 1], s=70,
                       facecolors="none" if hollow else PURPLE,
                       edgecolors=PURPLE, linewidths=1.2,
                       alpha=0.30, zorder=3, clip_on=False)
            mx, my = arr[:, 0].mean(), arr[:, 1].mean()
            means[scale] = (mx, my)
            env_means[scale].append((mx, my))
            ax.scatter([mx], [my], s=170,
                       facecolors="none" if hollow else PURPLE,
                       edgecolors=PURPLE, linewidths=1.8,
                       alpha=0.85, zorder=4, clip_on=False)
        (x1, y1), (x0, y0) = means["1.0"], means["0.0"]
        ax.annotate("", xy=(x0, y0), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.3,
                                    alpha=0.45, shrinkA=9, shrinkB=9))
        dx, dy = LABEL_DXY[env]
        # x axis is reversed; positive dx nudges the label leftward on screen.
        ax.annotate(ENV_LABEL[env], xy=(x0 + dx, y0 + dy), fontsize=13,
                    color="#5a4a76", ha="right", va="center", zorder=6)

    print(f'{"class":24s}  hack             retain')
    for scale, name, hollow in (("0.0", "deployed (fs 0.0)", False),
                                ("1.0", "pre-ablation (fs 1.0)", True)):
        pts = np.array(env_means[scale])
        xs, ys = pts[:, 0], pts[:, 1]
        n = len(xs)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        x_ci = tcrit * float(np.std(xs, ddof=1) / np.sqrt(n))
        y_ci = tcrit * float(np.std(ys, ddof=1) / np.sqrt(n))
        print(f"{name:24s}  {xs.mean():.3f} +/- {x_ci:.3f}  {ys.mean():.3f} +/- {y_ci:.3f}")
        eb = ax.errorbar(xs.mean(), ys.mean(), xerr=x_ci, yerr=y_ci, fmt="o",
                         markersize=19, color=PURPLE,
                         markerfacecolor="none" if hollow else PURPLE,
                         markeredgecolor=PURPLE if hollow else "white",
                         markeredgewidth=2.0 if hollow else 1.6, ecolor=PURPLE,
                         elinewidth=2.2, capsize=7, capthick=2.2, zorder=5)
        for artist in (a for a in (eb.lines[0], *eb.lines[1], *eb.lines[2]) if a is not None):
            artist.set_clip_on(False)

    ax.set_xlim(1.03, -0.03)   # reversed: lower hack rate to the right
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Unintended behavior frequency (better →)")
    ax.set_ylabel("Target task performance (better ↑)")
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=PURPLE,
               markerfacecolor=PURPLE, markeredgecolor="white",
               markeredgewidth=1.6, markersize=13,
               label="GR v2 +easy-anchor 1:16: deployed (fs 0.0)"),
        Line2D([], [], marker="o", linestyle="none", color=PURPLE,
               markerfacecolor="none", markeredgecolor=PURPLE,
               markeredgewidth=2.0, markersize=13,
               label="GR v2: pre-ablation (fs 1.0)"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=12.5)
    note = ("gr_v2suite pilot (hard-split eval, n=256/seed, 3 seeds); "
            "arrows: env mean pre-ablation -> deployed; big: class mean, 95% t-CI over envs")
    fig.text(0.5, 0.005, note, ha="center", fontsize=11, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, "figs", f"v2suite_scatter_7envs.{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
