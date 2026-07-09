"""Countdown-Code endpoint comparison: GT hack rate vs retain, mean over 3 seeds.

Arms (all step-200 endpoints; GT hack = hack_freq = R_proxy & !R_true, i.e. ANY
successful hack, not just monitored):
  - do-nothing, RP=2, RP=5           : policy level (retain=1, forget=1)
  - GR+coherence (round 2)           : deployed (fs0.0) AND both-adapter (fs1.0)
  - GR no-coherence, per-seed optimal: both-adapter config at the highest forget
    scale whose MONITORED hack rate < the base model's monitored rate (5/256);
    0.0 if no scale qualifies. (Selection uses the training classifier only.)
  - SFT base reference (no RL), binomial CI.

Measurement protocol: posthoc fseval (step-200 ckpt, n_eval=256, HF bf16, 1536
cap) everywhere EXCEPT two runs whose 4h-timeout deaths left no checkpoint
(round-2 do-nothing s15, endpoint ~step 144; round-2 GR+coh s15, ~step 195):
those use the mean of their last 3 in-training evals (n=64/mode) — the two
protocols were verified to agree on coh-s9.

Error bars: 95% t-interval for the MEAN over the 3 seeds (df=2, t=4.303), both
axes. Base point: 95% binomial (normal approx) on n=256.

With --step N ("early stopping"): every arm is measured from the in-training
evals instead (mean of the last 3 evals at steps <= N, n=64/mode each; no
checkpoints exist mid-run), the no-coherence arm is omitted, and the figure is
labeled accordingly. The SFT-base reference point is step-independent.

Usage: .venv/bin/python figures_pareto/countdown_arms_scatter.py [--step 100]
"""
import argparse
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

NOCOH_DIR = os.path.join(ROOT, "output", "countdown_code_gr_nocoh-0703-0154_fseval")
COH_DIR = os.path.join(ROOT, "output", "countdown_code_gr-0702-0134_fseval")
COH64_DIR = os.path.join(ROOT, "output", "countdown_code_gr_coh64_fseval")
RP2_DIR = os.path.join(ROOT, "output", "countdown_code_rp2-0702-0026_fseval")

BASE_MON = 5 / 256  # SFT-base monitored (detected) hack rate, from the base anchor eval

T95_DF2 = 4.30265  # t(0.975, df=2): 95% CI for a mean of 3 seeds


def _val(d, prefix):
    hits = [v for k, v in d.items() if k.startswith(prefix) and "/" not in k[len(prefix):]]
    assert len(hits) == 1, f"prefix {prefix!r}: {len(hits)} hits"
    return hits[0]


def fs_point(path, scale):
    """(gt_hack, retain, monitored) at a forget scale from an fseval JSON."""
    d = json.load(open(path))["scales"][scale]
    return _val(d, "hack_freq/"), _val(d, "retain/"), _val(d, "hack_freq_detectable/")


def intraining_last3(path, mode, at_step=None):
    """(gt_hack, retain) = mean of the last 3 in-training evals for a mode,
    restricted to evals at steps <= at_step when given (early stopping: no
    lookahead past the stop point)."""
    rows = [json.loads(l) for l in open(path)]
    by_step = {}
    for r in rows:
        by_step[r["step"]] = r  # dedup keep-last (restart guidance)
    steps = [s for s in sorted(by_step) if at_step is None or s <= at_step]
    assert len(steps) >= 3, f"{path}: only {len(steps)} evals at steps <= {at_step}"
    last3 = [by_step[s] for s in steps[-3:]]
    def g(r, pref):
        return next(v for k, v in r.items() if k.startswith(f"{mode}/{pref}"))
    return (st.mean(g(r, "hack_freq/") for r in last3),
            st.mean(g(r, "retain/") for r in last3))


def optimal_scale(path):
    """Highest forget scale with monitored < BASE_MON; '0.0' if none qualifies."""
    scales = json.load(open(path))["scales"]
    ok = [s for s in scales if _val(scales[s], "hack_freq_detectable/") < BASE_MON]
    return max(ok, key=float) if ok else "0.0"


def arm_stats(points):
    """points: list of (gt, retain) per seed -> means, SEM, and 95% t-CI."""
    gts, rets = [p[0] for p in points], [p[1] for p in points]
    n = len(points)
    assert n == 3, f"expected 3 seeds, got {n}"
    sem = lambda vs: st.stdev(vs) / math.sqrt(n)
    return {"gt": st.mean(gts), "gt_sem": sem(gts), "gt_ci": T95_DF2 * sem(gts),
            "ret": st.mean(rets), "ret_sem": sem(rets), "ret_ci": T95_DF2 * sem(rets),
            "per_seed": points}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=None,
                    help="early stopping: measure all arms from in-training evals at steps <= N")
    args = ap.parse_args()

    coh = {s: os.path.join(COH_DIR, f"countdown_code_gr_cls_coh256_pen2_noretain_balanced_splitmoment_lam1_s{s}.json")
           for s in (9, 16)}
    coh64 = {s: os.path.join(COH64_DIR, f"countdown_code_gr_cls_coh64_pen2_noretain_balanced_splitmoment_lam1_s{s}.json")
             for s in (9, 15, 16)}
    rp = lambda amt, s: os.path.join(
        RP2_DIR, f"reward_penalty_countdown_code_hack_reward_penalty_amount{amt}_s{s}.json")
    dn = lambda s: os.path.join(RP2_DIR, f"countdown_code_hack_reward_penalty_amountmissing_s{s}.json")

    def reval(sweep, run):
        return os.path.join(ROOT, "output", sweep, run, "routing_eval.jsonl")

    dn_run = lambda s: reval("countdown_code_rp2-0702-0026",
                             f"countdown_code_hack_reward_penalty_amountmissing_s{s}")
    rp_run = lambda amt, s: reval("countdown_code_rp2-0702-0026",
                                  f"reward_penalty_countdown_code_hack_reward_penalty_amount{amt}_s{s}")
    coh_run = lambda s: reval("countdown_code_gr-0702-0134",
                              f"countdown_code_gr_cls_coh256_pen2_noretain_balanced_splitmoment_lam1_s{s}")
    coh64_run = lambda s: reval("countdown_code_gr_coh64",
                                f"countdown_code_gr_cls_coh64_pen2_noretain_balanced_splitmoment_lam1_s{s}")

    arms = {}

    if args.step is not None:
        # Early stopping: all arms from in-training evals at steps <= N.
        N = args.step
        arms["do-nothing"] = arm_stats([intraining_last3(dn_run(s), "both", N) for s in (9, 15, 16)])
        arms["RP=2"] = arm_stats([intraining_last3(rp_run("2.0", s), "both", N) for s in (9, 15, 16)])
        arms["RP=5"] = arm_stats([intraining_last3(rp_run("5.0", s), "both", N) for s in (9, 15, 16)])
        arms["GR coh 1:4 (deployed)"] = arm_stats(
            [intraining_last3(coh_run(s), "retain_only", N) for s in (9, 15, 16)])
        arms["GR coh 1:4 (both)"] = arm_stats(
            [intraining_last3(coh_run(s), "both", N) for s in (9, 15, 16)])
        arms["GR coh 1:16 (deployed)"] = arm_stats(
            [intraining_last3(coh64_run(s), "retain_only", N) for s in (9, 15, 16)])
        arms["GR coh 1:16 (both)"] = arm_stats(
            [intraining_last3(coh64_run(s), "both", N) for s in (9, 15, 16)])
    else:
        arms["do-nothing"] = arm_stats(
            [fs_point(dn(9), "1.0")[:2], intraining_last3(dn_run(15), "both"),
             fs_point(dn(16), "1.0")[:2]])
        arms["RP=2"] = arm_stats([fs_point(rp("2.0", s), "1.0")[:2] for s in (9, 15, 16)])
        arms["RP=5"] = arm_stats([fs_point(rp("5.0", s), "1.0")[:2] for s in (9, 15, 16)])

        coh_s15_dep = intraining_last3(coh_run(15), "retain_only")
        coh_s15_both = intraining_last3(coh_run(15), "both")
        arms["GR coh 1:4 (deployed)"] = arm_stats(
            [fs_point(coh[9], "0.0")[:2], coh_s15_dep, fs_point(coh[16], "0.0")[:2]])
        arms["GR coh 1:4 (both)"] = arm_stats(
            [fs_point(coh[9], "1.0")[:2], coh_s15_both, fs_point(coh[16], "1.0")[:2]])
        arms["GR coh 1:16 (deployed)"] = arm_stats(
            [fs_point(coh64[s], "0.0")[:2] for s in (9, 15, 16)])
        arms["GR coh 1:16 (both)"] = arm_stats(
            [fs_point(coh64[s], "1.0")[:2] for s in (9, 15, 16)])

        # GR no-coherence at the MONITORED-pick operating point (re-added
        # 2026-07-07): per-seed forget scale = highest with monitored hack rate
        # < the base model's monitored rate (optimal_scale; fallback '0.0' when
        # no scale qualifies — which is the s15 drift seed), plotted at the GT
        # hack rate. Classifier-only selection: no ground truth in the pick.
        nocoh = {s: os.path.join(NOCOH_DIR,
                                 f"countdown_code_gr_nocoh_cls_noretain_balanced_splitmoment_lam1_s{s}.json")
                 for s in (9, 15, 16)}
        nocoh_points, nocoh_scales = [], {}
        for s in (9, 15, 16):
            sc = optimal_scale(nocoh[s])
            nocoh_scales[s] = sc
            nocoh_points.append(fs_point(nocoh[s], sc)[:2])
        arms["GR no-coh (fs by monitor)"] = arm_stats(nocoh_points)
        print(f"no-coh monitored-pick scales: {nocoh_scales}")

    base = json.load(open(os.path.join(
        NOCOH_DIR, "countdown_code_gr_nocoh_cls_noretain_balanced_splitmoment_lam1_s9__r0.0.json")))
    bd = base["scales"]["0.0"]
    base_gt, base_ret = _val(bd, "hack_freq/"), _val(bd, "retain/")
    binom = lambda p: 1.96 * math.sqrt(p * (1 - p) / 256)

    print(f"\n{'arm':<24} {'GT hack':>8} {'±SEM':>6} {'retain':>7} {'±SEM':>6}   per-seed (gt, ret)")
    for name, a in arms.items():
        ps = "  ".join(f"({g:.3f},{r:.3f})" for g, r in a["per_seed"])
        print(f"{name:<24} {a['gt']:>8.3f} {a['gt_sem']:>6.3f} {a['ret']:>7.3f} {a['ret_sem']:>6.3f}   {ps}")
    print(f"{'SFT base (no RL)':<24} {base_gt:>8.3f} {binom(base_gt):>6.3f} {base_ret:>7.3f} {binom(base_ret):>6.3f}   (binomial CI, n=256)")

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STYLE = {  # arm -> (color, marker, filled); 1:4 dose = blues, 1:16 = greens
        "do-nothing":             ("#52514e", "X", True),
        "RP=2":                   ("#eda100", "s", True),
        "RP=5":                   ("#eb6834", "D", True),
        "GR coh 1:4 (deployed)":  ("#2a78d6", "o", True),
        "GR coh 1:4 (both)":      ("#86b6ef", "o", True),
        "GR coh 1:16 (deployed)": ("#008300", "^", True),
        "GR coh 1:16 (both)":     ("#1baf7a", "^", True),
        "GR no-coh (fs by monitor)": ("#4a3aa7", "v", True),
    }
    # Direct labels only where uncrowded; the legend carries full identity.
    if args.step is not None:  # offsets tuned to the step-100 point geometry
        LABEL_OFF = {
            "do-nothing":             (10, 12, "left"),
            "GR coh 1:4 (deployed)":  (0, -14, "center"),
            "GR coh 1:16 (deployed)": (0, 13, "center"),
        }
    else:  # tuned to endpoint positions
        LABEL_OFF = {
            "do-nothing":             (10, 10, "left"),
            "GR coh 1:4 (deployed)":  (12, -1, "left"),
            "GR coh 1:16 (deployed)": (0, 13, "center"),
            "GR no-coh (fs by monitor)": (0, -14, "center"),
        }

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for name, a in arms.items():
        color, marker, _ = STYLE[name]
        for g, r in a["per_seed"]:
            ax.plot(g, r, marker, color=color, ms=5, alpha=0.35,
                    markeredgewidth=0, zorder=2)
        ax.errorbar(a["gt"], a["ret"], xerr=a["gt_sem"], yerr=a["ret_sem"],
                    fmt=marker, color=color, ms=9, elinewidth=1.2, capsize=3,
                    zorder=3, label=name)
        if name in LABEL_OFF:
            dx, dy, ha = LABEL_OFF[name]
            ax.annotate(name, (a["gt"], a["ret"]), xytext=(dx, dy),
                        textcoords="offset points", fontsize=9, color=color,
                        va="center", ha=ha)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.plot(base_gt, base_ret, "o", mfc="none", color="#8a8984", ms=9, zorder=2)
    ax.annotate("SFT base (no RL)", (base_gt, base_ret), xytext=(0, 13),
                textcoords="offset points", fontsize=9, color="#8a8984",
                va="center", ha="center")

    ax.set_xlabel("ground-truth hack rate  (any successful hack; better →)")
    ax.set_ylabel("retain: true_tested rate  (better ↑)")
    if args.step is not None:
        ax.set_title(f"Countdown-Code, early stopping @ step {args.step} — mean ± SEM over 3 seeds "
                     "(faint = per-seed)\n(monitor = test_overwrite only; per seed = mean of the "
                     f"evals @ steps {args.step - 20}/{args.step - 10}/{args.step}, n=64/mode each)",
                     fontsize=10)
    else:
        ax.set_title("Countdown-Code @ 200 steps — mean ± SEM over 3 seeds (faint = per-seed)\n"
                     "(monitor = test_overwrite only)",
                     fontsize=10)
    # Rates live in [0,1]; huge t-CIs (n=3, heterogeneous seeds) clip at the edges.
    ax.set_xlim(1.02, -0.02)   # inverted: lower hack rate to the right
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="#e5e4e0", lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if args.step is None:
        fig.text(0.995, 0.005,
                 "do-nothing s15 (step~144) and GR+coh s15 (step~195): no ckpt (4h timeout) — in-training eval used",
                 ha="right", fontsize=7, color="#8a8984")
    fig.tight_layout()
    suffix = f"_step{args.step}" if args.step is not None else ""
    out = os.path.join(HERE, "figs", f"countdown_arms_scatter{suffix}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=180)
    print(f"\nwrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
