"""Per-env probe-metric trajectories: x = checkpoint step, rows = envs, cols = metrics.

Cols: (1) Lambda at hack_onset (undetected) vs clean_retain baseline, IQR bands
      (2) signed alignment s_R / s_F at hack onsets (the tug-of-war)
      (3) raw rho_R / rho_F at hack onsets (Lambda's components, log scale)
      (4) per-token loss at hack onsets + clean tokens (saturation context)
Reads probe_data/<env>_probe_agg.json. Out: paper_figures/probe_metrics.{png,pdf}
"""
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sorted(glob.glob("probe_data/*_probe_agg.json"))
files = [f for f in files if "/excl_" not in f and "/addseed_" not in f]
envs = [f.split("/")[-1].replace("_probe_agg.json", "") for f in files]
# addition_v2 is bimodal across seeds: show one representative row per modality.
# s1 = conditional/failed-localization (the default row), s3 = unconditional/localized.
i = envs.index("addition_v2")
files.insert(i + 1, "probe_data/addseed_s3_probe_agg.json")
envs[i] = "addition_v2 s1\n(conditional)"
envs.insert(i + 1, "addition_v2 s3\n(localized)")
SEED_OVERRIDE = {"addition_v2 s1\n(conditional)": ("addition_v2", "_s1"),
                 "addition_v2 s3\n(localized)": ("addition_v2", "_s3")}
fig, axes = plt.subplots(len(envs) + 1, 6, figsize=(27, 3.1 * (len(envs) + 1)), squeeze=False)

RUN_GLOB = "output/retrain_gr_modal_all_classic_nocoh_canonical_steps/{env}_*_s1"


def hack_freq_series(env):
    if env in SEED_OVERRIDE:
        base, seed = SEED_OVERRIDE[env]
        runs = [r for r in glob.glob(RUN_GLOB.format(env=base).replace("_s1", seed))]
    else:
        runs = glob.glob(RUN_GLOB.format(env=env))
    if not runs:
        return [], [], None
    seen = {}
    for line in open(runs[0] + "/routing_eval.jsonl"):
        e = json.loads(line)
        key = next((k for k in e if k.startswith("both/hack_freq")), None)
        if key:
            seen[e["step"]] = e[key]
    xs = sorted(seen)
    ys = [seen[x] for x in xs]
    # emergence = first step where hack rate exceeds the step-0 rate by 5pp absolute
    base = ys[0] if ys else 0.0
    onset = next((x for x, y in zip(xs, ys) if y >= base + 0.05), None)
    return xs, ys, onset


def cell(d, step, cls, traj):
    for c in d["cells"]:
        if (c["step"], c["cls"], c["traj"]) == (step, cls, traj):
            return c
    return None


for ei, (f, env) in enumerate(zip(files, envs)):
    d = json.load(open(f))
    steps = sorted({c["step"] for c in d["cells"]})

    def series(cls, traj, key):
        med, lo, hi, xs = [], [], [], []
        for st in steps:
            c = cell(d, st, cls, traj)
            if c and key in c:
                xs.append(st); med.append(c[key]["med"])
                lo.append(c[key]["q25"]); hi.append(c[key]["q75"])
        return xs, med, lo, hi

    ax = axes[ei][0]
    for cls, traj, color, label in [("hack_onset", "undetected", "#d62728", "hack onset (blind-spot)"),
                                    ("hack_onset", "detected", "#ff9896", "hack onset (detected)"),
                                    ("clean_retain", "clean", "#7f7f7f", "clean tokens")]:
        xs, m, lo, hi = series(cls, traj, "lam")
        if xs:
            share = lambda v: [x / (1 + x) for x in v]
            ax.plot(xs, share(m), color=color, lw=2, label=label)
            ax.fill_between(xs, share(lo), share(hi), color=color, alpha=0.15)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{env}\nρ_F / (ρ_R + ρ_F)")
    if ei == 0:
        ax.set_title("forget share of gradient norm"); ax.legend(fontsize=7)

    ax = axes[ei][1]
    for cls, traj, color, label in [("hack_onset", "undetected", "#d62728", "hack onset (blind-spot)"),
                                    ("hack_onset", "detected", "#ff9896", "hack onset (detected)"),
                                    ("clean_retain", "clean", "#7f7f7f", "clean tokens")]:
        xs, m, lo, hi = series(cls, traj, "lam")
        if xs:
            ax.plot(xs, m, color=color, lw=2, label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_ylabel("Λ = ρ_F/ρ_R")
    if ei == 0:
        ax.set_title("gradient preference Λ"); ax.legend(fontsize=7)

    ax = axes[ei][2]
    for key, color, label in [("s_r", "#2ca02c", "ŝ_R hack-onset"), ("s_f", "#9467bd", "ŝ_F hack-onset")]:
        xs, m, lo, hi = series("hack_onset", "undetected", key)
        if xs:
            ax.plot(xs, m, color=color, lw=2, label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    for key, color, label in [("s_r", "#2ca02c", "ŝ_R clean"), ("s_f", "#9467bd", "ŝ_F clean")]:
        xs, m, lo, hi = series("clean_retain", "clean", key)
        if xs:
            ax.plot(xs, m, color=color, lw=1.4, ls="--", alpha=0.7, label=label)
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set_ylabel("signed alignment ŝ")
    if ei == 0:
        ax.set_title("co-implement (+) vs counteract (−): solid=hack onsets, dashed=clean"); ax.legend(fontsize=6)

    ax = axes[ei][3]
    for key, color, label in [("rho_r", "#2ca02c", "ρ_R hack-onset"), ("rho_f", "#9467bd", "ρ_F hack-onset")]:
        xs, m, lo, hi = series("hack_onset", "undetected", key)
        if xs:
            ax.plot(xs, m, color=color, lw=2, label=label)
    for key, color, label in [("rho_r", "#2ca02c", "ρ_R clean"), ("rho_f", "#9467bd", "ρ_F clean")]:
        xs, m, lo, hi = series("clean_retain", "clean", key)
        if xs:
            ax.plot(xs, m, color=color, lw=1.4, ls="--", alpha=0.7, label=label)
    ax.set_ylabel("ρ (norm. grad)")
    if ei == 0:
        ax.set_title("raw receptivity: solid=hack onsets, dashed=clean"); ax.legend(fontsize=6)

    ax = axes[ei][4]
    for cls, traj, color, label in [("hack_onset", "undetected", "#d62728", "hack onset"),
                                    ("hack_onset", "detected", "#ff9896", "hack onset (detected)"),
                                    ("clean_retain", "clean", "#7f7f7f", "clean")]:
        xs, m, lo, hi = series(cls, traj, "loss")
        if xs:
            ax.plot(xs, m, color=color, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("per-token CE loss")
    if ei == 0:
        ax.set_title("saturation context"); ax.legend(fontsize=7)

for ei, env in enumerate(envs):
    xs, ys, onset = hack_freq_series(env)
    ax = axes[ei][5]
    if xs:
        ax.plot(xs, ys, color="#1f77b4", lw=2)
        ax.set_ylim(0, 1)
    ax.set_ylabel("hack frequency (both)")
    if ei == 0:
        ax.set_title("behavior emergence (run's own eval)")
    if onset is not None:
        for ax2 in axes[ei]:
            ax2.axvline(onset, color="k", ls=":", lw=1.2, alpha=0.7)

# ---- aggregate row: mean over envs (excluding the conditional addition_v2 s1 row) ----
AGG_EXCLUDE = {"probe_data/addition_v2_probe_agg.json"}
GRID = np.linspace(0.05, 1.0, 20)


def norm_series(fpath, cls, traj, key, transform=None):
    d = json.load(open(fpath))
    xs, ms = [], []
    steps = sorted({c["step"] for c in d["cells"]})
    if not steps:
        return None
    mx = max(steps)
    for c in sorted(d["cells"], key=lambda c: c["step"]):
        if (c["cls"], c["traj"]) == (cls, traj) and key in c:
            xs.append(c["step"] / mx); ms.append(c[key]["med"])
    if len(xs) < 3:
        return None
    v = np.interp(GRID, xs, ms)
    return transform(v) if transform else v


def agg(cls, traj, key, transform=None):
    vs = [norm_series(f, cls, traj, key, transform) for f in files if f not in AGG_EXCLUDE]
    vs = [v for v in vs if v is not None]
    return np.mean(vs, axis=0) if vs else None


AR = len(envs)
share_t = lambda v: v / (1 + v)
for ax in axes[AR]:
    ax.set_facecolor("#f2f2f2")

panel_specs = [
    (0, [("hack_onset", "undetected", "lam", share_t, "#d62728", "-"),
         ("clean_retain", "clean", "lam", share_t, "#7f7f7f", "-")], "MEAN\nρ_F/(ρ_R+ρ_F)", (0, 1), 0.5),
    (1, [("hack_onset", "undetected", "lam", None, "#d62728", "-"),
         ("clean_retain", "clean", "lam", None, "#7f7f7f", "-")], "Λ", None, 1.0),
    (2, [("hack_onset", "undetected", "s_r", None, "#2ca02c", "-"),
         ("hack_onset", "undetected", "s_f", None, "#9467bd", "-"),
         ("clean_retain", "clean", "s_r", None, "#2ca02c", "--"),
         ("clean_retain", "clean", "s_f", None, "#9467bd", "--")], "ŝ", None, 0.0),
    (3, [("hack_onset", "undetected", "rho_r", None, "#2ca02c", "-"),
         ("hack_onset", "undetected", "rho_f", None, "#9467bd", "-"),
         ("clean_retain", "clean", "rho_r", None, "#2ca02c", "--"),
         ("clean_retain", "clean", "rho_f", None, "#9467bd", "--")], "ρ", None, None),
    (4, [("hack_onset", "undetected", "loss", None, "#d62728", "-"),
         ("clean_retain", "clean", "loss", None, "#7f7f7f", "-")], "CE loss", None, None),
]
for col, lines, ylab, ylim, ref in panel_specs:
    ax = axes[AR][col]
    for cls, traj, key, tr, color, ls in lines:
        v = agg(cls, traj, key, tr)
        if v is not None:
            ax.plot(GRID, v, color=color, ls=ls, lw=2.2)
    if ref is not None:
        ax.axhline(ref, color="k", ls=":", lw=1)
    if ylim:
        ax.set_ylim(*ylim)
    if col == 4:
        ax.set_yscale("log")
    ax.set_ylabel(ylab)
    ax.set_xlabel("fraction of training")

# aggregate hack frequency + mean normalized emergence
GRID_HF = np.linspace(0.0, 1.0, 200)  # hack-freq evals are ~10x denser than checkpoints
hfs, onsets = [], []
for env in envs:
    if env.startswith("addition_v2 s1"):
        continue
    xs, ys, onset = hack_freq_series(env)
    if xs:
        mx = max(xs)
        hfs.append(np.interp(GRID_HF, [x / mx for x in xs], ys))
        if onset is not None:
            onsets.append(onset / mx)
ax = axes[AR][5]
if hfs:
    ax.plot(GRID_HF, np.mean(hfs, axis=0), color="#1f77b4", lw=2.2)
    ax.set_ylim(0, 1)
ax.set_ylabel("hack frequency")
ax.set_xlabel("fraction of training")
if onsets:
    for ax2 in axes[AR]:
        ax2.axvline(np.mean(onsets), color="k", ls=":", lw=1.2, alpha=0.7)

for ax in axes[-2]:
    ax.set_xlabel("training step")
fig.suptitle("Gradient-probe trajectories — classic routing, no coherence (1 seed/env; "
             "median + IQR over probe positions)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
_y0 = axes[len(envs)][0].get_position().y1
_y1 = axes[len(envs) - 1][0].get_position().y0
_ysep = (_y0 + _y1) / 2
import matplotlib.lines as mlines
fig.add_artist(mlines.Line2D([0.02, 0.98], [_ysep, _ysep], color="k", lw=1.5,
                             transform=fig.transFigure))
for ext in ("png", "pdf"):
    fig.savefig(f"paper_figures/probe_metrics.{ext}", dpi=130, bbox_inches="tight")
print(f"saved paper_figures/probe_metrics.png ({len(envs)} envs)")
