"""Slide version of the classic-routing probe figure (v5: two separate figures).

probe_slide_envs: 2 rows x 7 cols, one column per env.
probe_slide_mean: 2 rows x 1 col, the cross-env mean (normalized x).
Top row = gradient-mass share (hack red / clean blue, 0.5 centered, 0.30-0.70).
Bottom row = s-hat (retain green / forget purple; solid hack, dashed clean; 0 centered, +-0.25).
addition_v2 = localized seed s3. Emergence vline = step-0 hack rate + 5pp; x origin offset so
early vlines stay visible.
"""
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 13, "xtick.labelsize": 11.5, "ytick.labelsize": 12,
})
FS_TITLE, FS_YLABEL, FS_XLABEL, FS_LEGEND = 15, 14, 13, 11

CELLS = [
    ("addition_v2 (s3)", "probe_data/addseed_s3_probe_agg.json",
     "output/retrain_gr_modal_all_classic_nocoh_canonical_steps/addition_v2_*_s3"),
    ("cities_qa", "probe_data/cities_qa_probe_agg.json",
     "output/retrain_gr_modal_all_classic_nocoh_canonical_steps/cities_qa_*_s1"),
    ("object_qa", "probe_data/object_qa_probe_agg.json",
     "output/retrain_gr_modal_all_classic_nocoh_canonical_steps/object_qa_*_s1"),
    ("persona_qa", "probe_data/persona_qa_probe_agg.json",
     "output/retrain_gr_modal_all_classic_nocoh_canonical_steps/persona_qa_*_s1"),
    ("repeat_extra", "probe_data/repeat_extra_probe_agg.json",
     "output/retrain_gr_modal_all_classic_nocoh_canonical_steps/repeat_extra_*_s1"),
    ("sorting_copy", "probe_data/sorting_copy_probe_agg.json",
     "output/retrain_gr_modal_all_classic_nocoh_canonical_steps/sorting_copy_*_s1"),
    ("topic_contains", "probe_data/topic_contains_probe_agg.json",
     "output/retrain_gr_modal_all_classic_nocoh_canonical_steps/topic_contains_*_s1"),
]
SHARE_YLIM = (0.30, 0.70)
SHAT_YLIM = (-0.25, 0.25)
RED, BLUE, GREEN, PURPLE = "#d62728", "#1f77b4", "#2ca02c", "#9467bd"
GRID = np.linspace(0.05, 1.0, 20)
share_t = lambda x: x / (1 + x)


def series(d, cls, traj, key, transform=None):
    xs, med, lo, hi = [], [], [], []
    for c in sorted(d["cells"], key=lambda c: c["step"]):
        if (c["cls"], c["traj"]) == (cls, traj) and key in c:
            xs.append(c["step"])
            med.append(c[key]["med"]); lo.append(c[key]["q25"]); hi.append(c[key]["q75"])
    if transform:
        med, lo, hi = ([transform(x) for x in v] for v in (med, lo, hi))
    return xs, med, lo, hi


def emergence(run_glob):
    runs = glob.glob(run_glob)
    if not runs:
        return None, None
    seen = {}
    for line in open(runs[0] + "/routing_eval.jsonl"):
        e = json.loads(line)
        key = next((k for k in e if k.startswith("both/hack_freq")), None)
        if key:
            seen[e["step"]] = e[key]
    xs = sorted(seen)
    ys = [seen[x] for x in xs]
    base = ys[0] if ys else 0.0
    onset = next((x for x, y in zip(xs, ys) if y >= base + 0.05), None)
    return onset, max(xs) if xs else None


def norm_series(agg_path, cls, traj, key, transform=None):
    d = json.load(open(agg_path))
    steps = sorted({c["step"] for c in d["cells"]})
    if not steps:
        return None
    mx = max(steps)
    xs, ms = [], []
    for c in sorted(d["cells"], key=lambda c: c["step"]):
        if (c["cls"], c["traj"]) == (cls, traj) and key in c:
            xs.append(c["step"] / mx); ms.append(c[key]["med"])
    if len(xs) < 3:
        return None
    v = np.interp(GRID, xs, ms)
    return np.array([transform(x) for x in v]) if transform else v


def style_rows(axT, axB):
    axT.axhline(0.5, color="k", ls=":", lw=1)
    axB.axhline(0.0, color="k", ls=":", lw=1)


def label_first_col(axT, axB):
    axT.set_ylim(*SHARE_YLIM)
    axT.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    axB.set_ylim(*SHAT_YLIM)
    axB.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    axT.set_ylabel("gradient mass\nproportion", fontsize=FS_YLABEL)
    axB.set_ylabel("output alignment ŝ", fontsize=FS_YLABEL)


def add_legends(axT, axB, bottom_loc="lower right"):
    axT.legend(handles=[Line2D([], [], color=RED, lw=2.2, label="hack tokens"),
                        Line2D([], [], color=BLUE, lw=2.2, label="clean tokens")],
               loc="lower right", fontsize=FS_LEGEND, frameon=True, framealpha=0.9,
               borderpad=0.3, labelspacing=0.25, handlelength=1.2)
    axB.legend(handles=[Line2D([], [], color=GREEN, lw=2.2, label="retain ŝ"),
                        Line2D([], [], color=PURPLE, lw=2.2, label="forget ŝ"),
                        Line2D([], [], color="0.4", lw=1.2, ls="--", label="clean tokens"),
                        Line2D([], [], color="k", lw=1.2, ls=":", label="hack emergence")],
               loc=bottom_loc, fontsize=FS_LEGEND, frameon=True, framealpha=0.9,
               borderpad=0.3, labelspacing=0.25, handlelength=1.2)


# --------------------------- Figure 1: the 7 envs ---------------------------
fig, axes = plt.subplots(2, 7, figsize=(17.5, 5.8), sharey="row",
                         gridspec_kw=dict(hspace=0.14, wspace=0.10,
                                          left=0.065, right=0.995, top=0.91, bottom=0.13))

for col, (name, agg_path, run_glob) in enumerate(CELLS):
    d = json.load(open(agg_path))
    axT, axB = axes[0][col], axes[1][col]
    for cls, traj, color in [("hack_onset", "undetected", RED), ("clean_retain", "clean", BLUE)]:
        xs, m, lo, hi = series(d, cls, traj, "lam", share_t)
        if xs:
            axT.plot(xs, m, color=color, lw=2.2)
            axT.fill_between(xs, lo, hi, color=color, alpha=0.14)
    for cls, traj, ls, alpha in [("hack_onset", "undetected", "-", 1.0),
                                 ("clean_retain", "clean", "--", 0.5)]:
        for key, color in [("s_r", GREEN), ("s_f", PURPLE)]:
            xs, m, lo, hi = series(d, cls, traj, key)
            if xs:
                axB.plot(xs, m, color=color, ls=ls, lw=2.2 if ls == "-" else 1.3, alpha=alpha)
                if ls == "-":
                    axB.fill_between(xs, lo, hi, color=color, alpha=0.14)
    onset, mx = emergence(run_glob)
    for ax in (axT, axB):
        if onset is not None:
            ax.axvline(onset, color="k", ls=":", lw=1.3, alpha=0.8)
        if mx:
            nice = 2000 if mx > 1500 else 1000
            ax.set_xlim(-0.045 * nice, 1.03 * nice)
            ax.set_xticks([0, nice // 2, nice])
    style_rows(axT, axB)
    axT.set_title(name, fontsize=FS_TITLE, fontweight="bold")
    axT.set_xticklabels([])
    axB.set_xlabel("training step", fontsize=FS_XLABEL)

label_first_col(axes[0][0], axes[1][0])
add_legends(axes[0][0], axes[1][0])

for ext in ("png", "pdf"):
    fig.savefig(f"paper_figures/probe_slide_envs.{ext}", dpi=150)
print("saved paper_figures/probe_slide_envs.png")

# --------------------------- Figure 2: the mean -----------------------------
fig2, (axT, axB) = plt.subplots(2, 1, figsize=(5.2, 5.8), sharex=True,
                                gridspec_kw=dict(hspace=0.14, left=0.20, right=0.97,
                                                 top=0.91, bottom=0.13))

for cls, traj, color in [("hack_onset", "undetected", RED), ("clean_retain", "clean", BLUE)]:
    vs = [norm_series(p, cls, traj, "lam", share_t) for _, p, _ in CELLS]
    vs = [v for v in vs if v is not None]
    if vs:
        axT.plot(GRID, np.mean(vs, axis=0), color=color, lw=2.4)
for cls, traj, ls, alpha in [("hack_onset", "undetected", "-", 1.0),
                             ("clean_retain", "clean", "--", 0.5)]:
    for key, color in [("s_r", GREEN), ("s_f", PURPLE)]:
        vs = [norm_series(p, cls, traj, key) for _, p, _ in CELLS]
        vs = [v for v in vs if v is not None]
        if vs:
            axB.plot(GRID, np.mean(vs, axis=0), color=color, ls=ls,
                     lw=2.4 if ls == "-" else 1.3, alpha=alpha)
onsets = []
for _, _, rg in CELLS:
    onset, mx = emergence(rg)
    if onset is not None and mx:
        onsets.append(onset / mx)
for ax in (axT, axB):
    if onsets:
        ax.axvline(np.mean(onsets), color="k", ls=":", lw=1.3, alpha=0.8)
    ax.set_xlim(-0.045, 1.03)
    ax.set_xticks([0, 0.5, 1.0])
style_rows(axT, axB)
axT.set_title("mean (all envs)", fontsize=FS_TITLE, fontweight="bold")
axB.set_xlabel("fraction of training", fontsize=FS_XLABEL)
label_first_col(axT, axB)
add_legends(axT, axB, bottom_loc="lower left")

for ext in ("png", "pdf"):
    fig2.savefig(f"paper_figures/probe_slide_mean.{ext}", dpi=150)
print("saved paper_figures/probe_slide_mean.png")
