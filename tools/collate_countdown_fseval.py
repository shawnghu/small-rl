"""Collate countdown_code forget-scale evals into curves + a scale pick.

Reads output/<sweep>_fseval/*.json (written by
tools/modal_train_gr.py::eval_forget_scales), groups runs by condition (run name
with the _s<seed> suffix stripped), aggregates over seeds per forget scale, and:

  1. prints the full per-scale table (monitored hack / ground-truth hack /
     per-form channels / retain), mean over seeds with per-seed values;
  2. prints the scale-selection candidates: for each scale, whether the
     MONITORED hack rate (hack_freq_detectable = the training classifier,
     test_overwrite; NOT ground truth) is zero in all seeds / whose mean is
     below 1/n_eval;
  3. writes a per-condition figure: retain + hack channels vs forget scale
     (mean line, per-seed faint traces).

Usage:
    .venv/bin/python tools/collate_countdown_fseval.py \
        --fsdirs output/countdown_code_gr_nocoh-0703-0154_fseval [more dirs...] \
        --out figures_pareto/figs/countdown_fseval
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

# metric-key prefix -> short label. Keys in the fseval JSON are
# "<prefix>/<env-specific suffix>"; match on prefix and take the single hit.
CHANNELS = {
    "retain": "retain/",                                # true_tested (scale-0 comps contribute 0)
    "monitored": "hack_freq_detectable/",               # test_overwrite = training classifier
    "hack_gt": "hack_freq/",                            # ground truth: R_proxy & !R_true
    "cr": "hack_freq_constraint_relax/",
    "mi": "hack_freq_modify_input/",
    "other": "hack_freq_other/",
    "combined": "combined/",
}

# hack_freq/ is a prefix of hack_freq_*/; resolve by exact-prefix with the
# longest CHANNELS prefix that matches, so order does not matter.
def _val(scale_dict, prefix):
    hits = [k for k in scale_dict if k.startswith(prefix)
            and "/" not in k[len(prefix):]]
    if not hits:
        return None
    assert len(hits) == 1, f"ambiguous metric prefix {prefix!r}: {hits}"
    return scale_dict[hits[0]]


def _cond_and_seed(run_name):
    m = re.match(r"^(.*)_s(\d+)$", run_name)
    assert m, f"run name {run_name!r} does not end in _s<seed>"
    return m.group(1), int(m.group(2))


def load(fsdirs):
    """condition -> {seed: (scales_dict, n_eval, step)}"""
    out = defaultdict(dict)
    for d in fsdirs:
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        assert files, f"no fseval JSONs in {d}"
        for f in files:
            rec = json.load(open(f))
            cond, seed = _cond_and_seed(rec["run_name"])
            out[cond][seed] = (rec["scales"], rec["n_eval"], rec["step"])
    return out


def collate(seed_recs):
    """-> (scales_sorted, {channel: {scale: [per-seed values]}}, n_eval)"""
    n_evals = {n for _, n, _ in seed_recs.values()}
    assert len(n_evals) == 1, f"mixed n_eval across seeds: {n_evals}"
    scale_sets = [set(sc.keys()) for sc, _, _ in seed_recs.values()]
    scales = sorted(set.intersection(*scale_sets), key=float)
    vals = {ch: defaultdict(list) for ch in CHANNELS}
    for seed in sorted(seed_recs):
        sc, _, _ = seed_recs[seed]
        for s in scales:
            for ch, prefix in CHANNELS.items():
                v = _val(sc[s], prefix)
                if v is not None:
                    vals[ch][s].append(v)
    return scales, vals, n_evals.pop()


def _m(vlist):
    """mean or nan for channels the eval didn't emit (e.g. hack_freq_modify_input
    is absent from routing-eval output; mi hacks then show up in hack_gt - cr - other)."""
    return st.mean(vlist) if vlist else float("nan")


def report(cond, scales, vals, n_eval, seeds):
    print(f"\n=== {cond}  (seeds {seeds}, n_eval={n_eval}) ===")
    hdr = f"{'scale':>5} | {'monitored':>22} | {'hack_gt':>8} | {'cr':>6} | {'mi':>6} | {'retain':>20}"
    print(hdr)
    print("-" * len(hdr))
    for s in scales:
        mon = vals["monitored"][s]
        ret = vals["retain"][s]
        mon_str = f"{_m(mon):.3f} ({'/'.join(f'{v:.3f}' for v in mon)})"
        ret_str = f"{_m(ret):.3f} ({'/'.join(f'{v:.2f}' for v in ret)})"
        print(f"{s:>5} | {mon_str:>22} | {_m(vals['hack_gt'][s]):>8.3f} | "
              f"{_m(vals['cr'][s]):>6.3f} | {_m(vals['mi'][s]):>6.3f} | {ret_str:>20}")

    # Selection: monitored channel only (the method may not consult ground truth).
    zero_all = [s for s in scales if all(v == 0.0 for v in vals["monitored"][s])]
    below_1 = [s for s in scales if st.mean(vals["monitored"][s]) <= 1.0 / n_eval]
    pick_zero = max(zero_all, key=float) if zero_all else None
    pick_soft = max(below_1, key=float) if below_1 else None
    print(f"selection (monitored hack, training classifier):")
    print(f"  highest scale with 0 monitored hacks in ALL seeds : {pick_zero}"
          f"  (retain {st.mean(vals['retain'][pick_zero]):.3f})" if pick_zero is not None
          else "  no scale has 0 monitored hacks in all seeds")
    if pick_soft is not None and pick_soft != pick_zero:
        print(f"  highest scale with mean monitored <= 1/n_eval      : {pick_soft}"
              f"  (retain {st.mean(vals['retain'][pick_soft]):.3f}, "
              f"monitored {st.mean(vals['monitored'][pick_soft]):.4f})")
    return pick_zero


def figure(conds, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Validated palette (dataviz reference instance) + marker as the secondary
    # (non-color) identity encoding.
    SERIES = [
        ("retain",    "retain (true_tested)",          "#2a78d6", "o"),
        ("monitored", "monitored hack (classifier)",   "#e34948", "s"),
        ("hack_gt",   "hack (ground truth)",           "#eda100", "^"),
        ("cr",        "constraint_relax",              "#4a3aa7", "D"),
    ]
    n = len(conds)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.6), squeeze=False, sharey=True)
    for ax, (cond, (scales, vals, n_eval, seeds)) in zip(axes[0], conds.items()):
        x = [float(s) for s in scales]
        end_labels = []
        for ch, label, color, marker in SERIES:
            if not any(vals[ch][s] for s in scales):
                continue  # channel absent from this eval's output
            per_seed = list(zip(*[vals[ch][s] for s in scales]))  # seed -> curve
            for curve in per_seed:
                ax.plot(x, curve, color=color, alpha=0.25, lw=1.0, zorder=1)
            means = [st.mean(vals[ch][s]) for s in scales]
            ax.plot(x, means, color=color, marker=marker, lw=2.0, ms=6,
                    label=label, zorder=3)
            end_labels.append([label.split(" (")[0], color, means[-1]])
        # direct labels at the right end, nudged apart when curves converge
        end_labels.sort(key=lambda e: e[2])
        MIN_GAP = 0.045
        for i in range(1, len(end_labels)):
            if end_labels[i][2] - end_labels[i - 1][2] < MIN_GAP:
                end_labels[i][2] = end_labels[i - 1][2] + MIN_GAP
        for text, color, y in end_labels:
            ax.annotate(text, (x[-1], y), xytext=(6, 0),
                        textcoords="offset points",
                        color=color, fontsize=8, va="center")
        ax.set_xlabel("forget-adapter scale (retain scale = 1.0)")
        ax.set_title(f"{cond}\n(seeds {seeds}, n_eval={n_eval}, step-200 ckpt)", fontsize=9)
        ax.set_xlim(-0.02, 1.28)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, color="#e5e4e0", lw=0.6, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0][0].set_ylabel("rate on eval set")
    axes[0][0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=180)
    print(f"\nwrote {out_base}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdirs", nargs="+", required=True)
    ap.add_argument("--out", default="figures_pareto/figs/countdown_fseval")
    args = ap.parse_args()

    grouped = load(args.fsdirs)
    conds = {}
    for cond, seed_recs in sorted(grouped.items()):
        scales, vals, n_eval = collate(seed_recs)
        seeds = sorted(seed_recs)
        report(cond, scales, vals, n_eval, seeds)
        conds[cond] = (scales, vals, n_eval, seeds)
    figure(conds, args.out)


if __name__ == "__main__":
    main()
