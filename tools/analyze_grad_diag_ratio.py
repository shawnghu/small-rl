#!/usr/bin/env python
"""Two methodology questions on the grad-diag sweep:

(#2) How much do the ABSOLUTE per-sample norms vary across envs / seeds? (If by
     orders of magnitude, a shared absolute threshold can't classify across
     envs.)

(#3) Does a WITHIN-SAMPLE ratio R_i = forget-param-norm_i / retain-param-norm_i
     (scale-invariant) separate forget from retain samples better / more
     transferably than the raw norms? We report, per metric:
       - median R on forget vs retain samples, per env
       - AUC of R as a per-sample classifier, per env, vs AUC of the raw
         forget-param norm
       - cross-env threshold transfer: balanced accuracy at a SINGLE global
         threshold (best on pooled data) vs each env's own best threshold.
         If global ~= per-env-best, the ratio is env-invariant.

Pools seeds; uses late-training steps (>= --late-frac of max step) with enough
hacks. Whole-model norms.

Usage: python tools/analyze_grad_diag_ratio.py output/<sweep_dir>/ [--late-frac 0.5] [--min-hacks 10]
"""
import argparse
import json
import os
import glob
from collections import defaultdict

import numpy as np


def env_of(run):
    for m in ("_gr_cls", "_gr_excl", "_rp", "_graddiag"):
        if m in run:
            return run.split(m)[0]
    return run


def short(e):
    return (e.replace("_sycophancy_conditional", "").replace("_conditional", "")
             .replace("_flattery", "").replace("_3xreward", "").replace("_extra", ""))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    lab = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])[order]
    v = allv[order]
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[lab == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def best_bal_acc(values, labels, threshold=None):
    """If threshold None, find threshold maximizing balanced accuracy (classify
    value>thr as forget=1). Returns (bal_acc, threshold)."""
    values = np.asarray(values)
    labels = np.asarray(labels)
    pos, neg = values[labels == 1], values[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan"), threshold
    if threshold is not None:
        tpr = (pos > threshold).mean()
        tnr = (neg <= threshold).mean()
        return 0.5 * (tpr + tnr), threshold
    cands = np.unique(values)
    if cands.size > 400:
        cands = np.quantile(values, np.linspace(0, 1, 400))
    best, bt = -1, cands[0]
    for t in cands:
        ba = 0.5 * ((pos > t).mean() + (neg <= t).mean())
        if ba > best:
            best, bt = ba, t
    return best, bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--late-frac", type=float, default=0.5)
    ap.add_argument("--min-hacks", type=int, default=10)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.sweep_dir, "*", "grad_diag.jsonl")))
    by_env = defaultdict(list)
    for f in files:
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            by_env[env_of(os.path.basename(os.path.dirname(f)))].append(recs)

    # gather late-step per-sample (a_f,a_r,g_f,g_r,label) per env (seeds pooled)
    data = {}  # env -> dict of arrays
    for env, rbs in by_env.items():
        maxstep = max(r["step"] for recs in rbs for r in recs)
        cut = maxstep * (1 - args.late_frac)
        cols = defaultdict(list)
        for recs in rbs:
            for r in recs:
                if r["step"] < cut:
                    continue
                if sum(r["is_rh"]) < args.min_hacks:
                    continue
                rh = r["is_rh"]
                af, ar = r["act_whole_model"]["forget"], r["act_whole_model"]["retain"]
                gf, gr = r["whole_model"]["forget"], r["whole_model"]["retain"]
                for j, lab in enumerate(rh):
                    cols["af"].append(af[j]); cols["ar"].append(ar[j])
                    cols["gf"].append(gf[j]); cols["gr"].append(gr[j])
                    cols["lab"].append(lab)
        if cols["lab"]:
            data[env] = {k: np.array(v, float) for k, v in cols.items()}
    envs = sorted(data)
    assert envs, "no late-step data"

    # ---- (#2) absolute-scale variability across envs/seeds
    print("=" * 78)
    print("(#2) ABSOLUTE norm scale across envs (median over ALL late samples)")
    print(f"{'env':<14} {'act forget':>11} {'act retain':>11} {'grad forget':>12} {'grad retain':>12}")
    med = {}
    for e in envs:
        d = data[e]
        med[e] = (np.median(d["af"]), np.median(d["ar"]), np.median(d["gf"]), np.median(d["gr"]))
        print(f"{short(e):<14} {med[e][0]:>11.2e} {med[e][1]:>11.2e} {med[e][2]:>12.2e} {med[e][3]:>12.2e}")
    for ci, name in enumerate(["act forget", "act retain", "grad forget", "grad retain"]):
        vals = np.array([med[e][ci] for e in envs])
        print(f"   {name:<12} cross-env spread: {vals.max()/vals.min():.0f}x "
              f"(min {vals.min():.1e} max {vals.max():.1e})")

    # ---- (#3) within-sample ratio R = forget-param / retain-param
    for metric, fkey, rkey in (("activation", "af", "ar"), ("gradient", "gf", "gr")):
        print("\n" + "=" * 78)
        print(f"(#3) WITHIN-SAMPLE ratio R = {metric} forget-param / retain-param")
        print(f"{'env':<14} {'medR|forget':>12} {'medR|retain':>12} {'AUC(R)':>8} "
              f"{'AUC(rawF)':>10} {'envBestBA':>10} {'thr*':>9}")
        pooledR, pooledLab = [], []
        per_env = {}
        for e in envs:
            d = data[e]
            ok = (d[rkey] > 0) & (d[fkey] > 0)
            R = np.log2(d[fkey][ok] / d[rkey][ok])  # log-ratio (symmetric, scale-free)
            lab = d["lab"][ok]
            rawF = np.log2(d[fkey][ok])
            mr_f = np.median(R[lab == 1]); mr_r = np.median(R[lab == 0])
            a_R = auc(R[lab == 1], R[lab == 0])
            a_raw = auc(rawF[lab == 1], rawF[lab == 0])
            ba, thr = best_bal_acc(R, lab)
            per_env[e] = (R, lab, thr)
            pooledR.append(R); pooledLab.append(lab)
            print(f"{short(e):<14} {mr_f:>12.2f} {mr_r:>12.2f} {a_R:>8.3f} "
                  f"{a_raw:>10.3f} {ba:>10.3f} {thr:>9.2f}")
        # cross-env transfer: single global threshold
        pR = np.concatenate(pooledR); pL = np.concatenate(pooledLab)
        gba, gthr = best_bal_acc(pR, pL)
        print(f"   global-threshold transfer (thr*={gthr:.2f}, bal-acc on pool={gba:.3f}):")
        for e in envs:
            R, lab, ethr = per_env[e]
            ba_global, _ = best_bal_acc(R, lab, threshold=gthr)
            ba_env, _ = best_bal_acc(R, lab)
            drop = ba_env - ba_global
            mark = "   <-- transfers poorly" if drop > 0.07 else ""
            print(f"      {short(e):<14} BA@global={ba_global:.3f}  BA@env-best={ba_env:.3f}  "
                  f"(loss {drop:+.3f}){mark}")


if __name__ == "__main__":
    main()
