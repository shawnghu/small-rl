"""Skyroute figures: per-behavior 2x2 (rows classic top / exclusive bottom; cols behavior-rate left /
Skywork-reward right; x = train step) + one composite overview.

Lines:
  both adapters (1,1)   blue    } from in-flight routing_eval (vLLM, training gen config) -- always present
  retain-only   (1,0)   green   }   steps ~21..181
  forget-only   (0,1)   red     }
  optimal forget scale  orange  -- from post-hoc (/tmp/skyroute_optimal_results.json), ONLY if present
Base anchor (step 0 = base-model rate, dashed grey ref + step-0 point on every line) is added ONLY when
the post-hoc base cache (/tmp/skyroute_scale4.json) is present -- it must be generated with the SAME gen
config as training (rep_penalty/top_k), see tools/modal_train_gr.py::_skyroute_gen_score.

Bold = mean over the 3 seeds, thin = individual seeds. Out: paper_figures/skyroute_<behavior>.{png,pdf}
and paper_figures/skyroute_overview.png.
"""
import json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BEH = ["em_dash", "semicolon", "ordinal", "bold", "evidential", "purple", "intensifier"]
SEEDS = [42, 43, 44]
MODES = ["classic", "exclusive"]
C = {"both": "#1f77b4", "retain": "#2ca02c", "forget": "#d62728", "optimal": "#ff7f0e"}
LABEL = {"both": "both (1,1)", "retain": "retain-only (1,0)", "forget": "forget-only (0,1)",
         "optimal": "optimal forget scale"}

inflight = json.load(open("/tmp/skyroute_inflight.json"))
optimal = scale4 = None
if os.path.exists("/tmp/skyroute_optimal_results.json"):
    optimal = {o["run"].split("/")[-1]: o for o in json.load(open("/tmp/skyroute_optimal_results.json"))}
if os.path.exists("/tmp/skyroute_scale4.json"):
    scale4 = {r["run"].split("/")[-1]: r for r in json.load(open("/tmp/skyroute_scale4.json"))}
LINES = ["both", "retain", "forget"] + (["optimal"] if optimal else [])
ANCHOR = scale4 is not None
print(f"lines={LINES}  base-anchor={ANCHOR}")

rn = lambda b, m, s: f"skyroute_{b}_{m}_s{s}"


def mean_series(per_seed):
    agg = defaultdict(list)
    for steps, vals in per_seed:
        for x, v in zip(steps, vals):
            agg[int(x)].append(v)
    xs = sorted(agg)
    return xs, [float(np.mean(agg[x])) for x in xs]


def series_for(beh, mode):
    """line -> {'rate':[(steps,vals)/seed], 'rew':[...]}, with optional step-0 base anchor."""
    out = {k: {"rate": [], "rew": []} for k in LINES}
    for s in SEEDS:
        name = rn(beh, mode, s)
        inf = inflight[name]
        anc = ([0], [0])
        if ANCHOR:
            bm = scale4[name]["modes"]["base"]
            br, brew = bm["behavior_rate"], bm["skywork_reward"]
        for ln, hf_key, rew_key in [("both", "both_hf", "both_rew"), ("retain", "retain_hf", "retain_rew"),
                                    ("forget", "forget_hf", "forget_rew")]:
            st = inf["step"]; rate = [100 * x for x in inf[hf_key]]; rew = inf[rew_key]
            if ANCHOR:
                st = [0] + st; rate = [br] + rate; rew = [brew] + rew
            out[ln]["rate"].append((st, rate)); out[ln]["rew"].append((st, rew))
        if optimal:
            by = optimal[name]["by_step"]
            osteps = sorted(int(k) for k in by)
            orate = [by[str(k)]["behavior_rate"] for k in osteps]
            orew = [by[str(k)]["skywork_reward"] for k in osteps]
            if ANCHOR:
                osteps = [0] + osteps; orate = [br] + orate; orew = [brew] + orew
            out["optimal"]["rate"].append((osteps, orate)); out["optimal"]["rew"].append((osteps, orew))
    return out


def draw(ax, data, metric):
    for ln in LINES:
        for steps, vals in data[ln][metric]:
            ax.plot(steps, vals, color=C[ln], alpha=0.22, lw=0.9)
        mx, mv = mean_series(data[ln][metric])
        ax.plot(mx, mv, color=C[ln], lw=2.5)
    ax.grid(alpha=0.25); ax.set_xlim(left=0)


# --- per-behavior 2x2 ---
for beh in BEH:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    if ANCHOR:
        bre = np.mean([scale4[rn(beh, m, s)]["modes"]["base"]["behavior_rate"] for m in MODES for s in SEEDS])
        brw = np.mean([scale4[rn(beh, m, s)]["modes"]["base"]["skywork_reward"] for m in MODES for s in SEEDS])
    for ri, mode in enumerate(MODES):
        data = series_for(beh, mode)
        for ci, metric in enumerate(["rate", "rew"]):
            ax = axes[ri][ci]
            draw(ax, data, metric)
            if ANCHOR:
                ax.axhline(bre if metric == "rate" else brw, color="grey", ls="--", lw=1.0, alpha=0.7)
            ax.set_title(f"{mode} — {'behavior rate' if metric == 'rate' else 'Skywork reward'}", fontsize=11)
            ax.set_ylabel("rate (%)" if metric == "rate" else "Skywork reward")
            if ri == 1:
                ax.set_xlabel("train step")
    fig.legend([Line2D([], [], color=C[k], lw=2.5) for k in LINES], [LABEL[k] for k in LINES],
               loc="upper center", ncol=len(LINES), fontsize=10, bbox_to_anchor=(0.5, 0.99), frameon=False)
    note = "" if ANCHOR else "  (in-flight only; base anchor + optimal pending corrected post-hoc)"
    fig.suptitle(f"{beh}: gradient-routing isolation{note}", y=1.0, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(f"paper_figures/skyroute_{beh}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved paper_figures/skyroute_{beh}.png/.pdf")

# --- composite overview (7 rows x 4 cols) ---
fig, axes = plt.subplots(7, 4, figsize=(16, 21))
cols = ["classic — rate %", "classic — reward", "exclusive — rate %", "exclusive — reward"]
for bi, beh in enumerate(BEH):
    for mi, mode in enumerate(MODES):
        data = series_for(beh, mode)
        for j, metric in enumerate(["rate", "rew"]):
            ax = axes[bi][mi * 2 + j]
            draw(ax, data, metric)
            if bi == 0:
                ax.set_title(cols[mi * 2 + j], fontsize=10)
    axes[bi][0].annotate(beh, xy=(-0.3, 0.5), xycoords="axes fraction", rotation=90, va="center",
                         ha="center", fontsize=12, fontweight="bold")
fig.legend([Line2D([], [], color=C[k], lw=2.5) for k in LINES], [LABEL[k] for k in LINES],
           loc="upper center", ncol=len(LINES), fontsize=11, bbox_to_anchor=(0.5, 0.997))
title = "Skyroute: which Skywork-taught behaviors gradient routing isolates"
fig.suptitle(title + ("" if ANCHOR else "  (in-flight 3-line; step-21..181; base+optimal pending)"), y=1.0, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig("paper_figures/skyroute_overview.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved paper_figures/skyroute_overview.png")
