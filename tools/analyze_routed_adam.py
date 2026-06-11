"""Final analysis for the routed_adam_emdash experiment (RoutedAdam vs capacity-matched control).

Panels:
  1. em-dash rate (both adapters): routed (4 seeds) vs m32 control (4 seeds), 0->200
  2. Skywork RM reward (both): routed vs control
  3. routed runs by adapter mode: both / retain-only / forget-only (the localization readout)
Plus paired stats: routed-vs-control dynamics gap and the forget-retain asymmetry CI over the
late window.

Reads output/routed_adam_emdash/{skyradam_m64_emdash_tok,skynorr_m32_emdash}_s*/routing_eval.jsonl.
Out: paper_figures/routed_adam_emdash.{png,pdf}

Run: .venv/bin/python tools/analyze_routed_adam.py
"""
import json, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Two roots: the original sweep dir (holds the 2 controls that finished before the orchestrator
# crash) and the timestamped relaunch dir (sweep.py appends -MMDD-HHMM when the dir exists).
ROOTS = ["output/routed_adam_emdash", "output/routed_adam_emdash-0610-0208"]
DET = "em_dash_detector"
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
LATE = [120, 140, 160, 180, 200]


def load(prefix):
    out = {}
    for root in ROOTS:
        for f in sorted(glob.glob(f"{root}/{prefix}_s*/routing_eval.jsonl")):
            name = f.split("/")[-2]
            assert name not in out, f"duplicate run {name} across roots — stale dir?"
            es = [json.loads(l) for l in open(f) if l.strip()]
            if es:
                out[name] = es
    return out


routed = load("skyradam_m64_emdash_tok")
ctrl = load("skynorr_m32_emdash")
print(f"routed: {len(routed)} runs, ctrl: {len(ctrl)} runs")
for name, es in {**routed, **ctrl}.items():
    print(f"  {name}: evals at {[e['step'] for e in es]}")


def series(runs, key):
    agg = defaultdict(list)
    for es in runs.values():
        for e in es:
            if key(e) is not None:
                agg[e["step"]].append(key(e))
    xs = sorted(agg)
    return xs, [np.mean(agg[x]) for x in xs], agg


def tci(v):
    v = np.array(v, float); n = len(v); m = v.mean()
    if n < 2: return m, np.nan, np.nan
    se = v.std(ddof=1) / np.sqrt(n)
    return m, m - T[n - 1] * se, m + T[n - 1] * se


fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

# Panel 1+2: routed vs control, em-dash and reward
for ax, keyname, conv, ylab in [
    (axes[0], f"both/hack_freq/{DET}", 100.0, "em-dash rate (%)"),
    (axes[1], "both/rate/skywork_reward_v2", 1.0, "Skywork RM reward"),
]:
    for runs, color, label in [(routed, "#9467bd", "RoutedAdam token-routed (m64)"),
                               (ctrl, "#1f77b4", "no-routing control (m32)")]:
        for es in runs.values():
            ax.plot([e["step"] for e in es], [conv * e[keyname] for e in es],
                    color=color, alpha=0.3, lw=0.9)
        xs, ms, _ = series(runs, lambda e: e.get(keyname))
        ax.plot(xs, [conv * m for m in ms], color=color, lw=2.6, label=label)
    ax.set_xlabel("train step"); ax.set_ylabel(ylab); ax.grid(alpha=0.25)
axes[0].set_title("combined-model behavior: routed vs control")
axes[1].set_title("combined-model RM reward")
axes[0].legend(fontsize=9)

# Panel 3: routed runs by adapter mode
MODES = [("both", "#1f77b4", "both (1,1)"), ("retain_only", "#2ca02c", "retain-only (1,0)"),
         ("forget_only", "#d62728", "forget-only (0,1)")]
for mode, color, label in MODES:
    for es in routed.values():
        axes[2].plot([e["step"] for e in es],
                     [100 * e[f"{mode}/hack_freq/{DET}"] for e in es],
                     color=color, alpha=0.3, lw=0.9)
    xs, ms, _ = series(routed, lambda e, m=mode: e.get(f"{m}/hack_freq/{DET}"))
    axes[2].plot(xs, [100 * m for m in ms], color=color, lw=2.6, label=label)
axes[2].set_xlabel("train step"); axes[2].set_ylabel("em-dash rate (%)")
axes[2].set_title("RoutedAdam runs: where does the behavior live?")
axes[2].grid(alpha=0.25); axes[2].legend(fontsize=9)

fig.suptitle("RoutedAdam exclusive token routing vs capacity-matched no-routing "
             "(em-dash, recall 1.0; 4 seeds each; bold = mean)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("png", "pdf"):
    fig.savefig(f"paper_figures/routed_adam_emdash.{ext}", dpi=140, bbox_inches="tight")
print("saved paper_figures/routed_adam_emdash.png/.pdf")

# Stats
print("\n=== late window (steps 120-200) ===")


def late(runs, key):
    return {n: 100 * np.mean([e[key] for e in es if e["step"] in LATE]) for n, es in runs.items()}


r_both = late(routed, f"both/hack_freq/{DET}")
c_both = late(ctrl, f"both/hack_freq/{DET}")
m, lo, hi = tci(list(r_both.values()));  print(f"routed both em%:  {m:5.1f} CI [{lo:.1f},{hi:.1f}]")
m, lo, hi = tci(list(c_both.values()));  print(f"control both em%: {m:5.1f} CI [{lo:.1f},{hi:.1f}]")

r_ret = late(routed, f"retain_only/hack_freq/{DET}")
r_fog = late(routed, f"forget_only/hack_freq/{DET}")
d = [r_fog[n] - r_ret[n] for n in sorted(routed)]
m, lo, hi = tci(d)
print(f"routed forget-retain asymmetry: per-seed {[f'{x:+.1f}' for x in d]} "
      f"mean {m:+.1f}pp CI [{lo:+.1f},{hi:+.1f}]")

rwd_r = {n: np.mean([e["both/rate/skywork_reward_v2"] for e in es if e["step"] in LATE])
         for n, es in routed.items()}
rwd_c = {n: np.mean([e["both/rate/skywork_reward_v2"] for e in es if e["step"] in LATE])
         for n, es in ctrl.items()}
m, lo, hi = tci(list(rwd_r.values())); print(f"routed RM (both):  {m:+.2f} CI [{lo:+.2f},{hi:+.2f}]")
m, lo, hi = tci(list(rwd_c.values())); print(f"control RM (both): {m:+.2f} CI [{lo:+.2f},{hi:+.2f}]")
