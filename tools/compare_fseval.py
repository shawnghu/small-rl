"""Regression compare: OLD vs NEW GRAFT forget-scale eval (graft-routing vs graft-port).

Loads the per-run forget-scale eval JSONs from two _fseval dirs, aggregates over
seeds per env, and prints a side-by-side of the load-bearing GRAFT metrics at the
forget=1 (pre-ablation / two-adapter) and forget=0 (deployment / retain-only)
endpoints, plus the criterion-optimal (post-ablation) point and the
monitored/unmonitored conditional gap. Pure-local (reads JSONs; no GPU).

Usage:
    .venv/bin/python tools/compare_fseval.py --old graft_canon_7envs --new graft_canon_port
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

# metric prefix -> human label. The suffix after '/' is env-specific (e.g.
# flattery_any, qa_correct), so we match by prefix and take the single key.
PREFIXES = {
    "hack_hackable": "hack_freq_hackable/",
    "hack_det": "hack_freq_hackable_detectable/",
    "hack_undet": "hack_freq_hackable_undetectable/",
    "retain": "retain/",
}


def _val(scale_dict, prefix):
    for k, v in scale_dict.items():
        if k.startswith(prefix):
            return v
    return None


def _short(run_name):
    return run_name.split("_graft_lam1_s")[0].split("_noint_lam1_s")[0]


def load(sweep, cond="graft"):
    """env -> list of per-seed run dicts (each = {scale: {metric: val}})."""
    out = defaultdict(list)
    for f in sorted(glob.glob(f"output/{sweep}_fseval/*{cond}_lam1_s*.json")):
        rec = json.load(open(f))
        out[_short(rec["run_name"])].append(rec["scales"])
    return out


def agg(runs, scale, prefix):
    """mean over seeds of the metric at a forget scale."""
    vals = [_val(r.get(scale, {}), prefix) for r in runs]
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else float("nan")


def crit_optimal(runs):
    """Per the monitored pareto: pick the forget scale maximizing retain - 2*hack
    (criterion-optimal 'post-ablation' point), then return (hack_det, hack_undet,
    retain) at it, mean over seeds. Hack = hack_freq_hackable."""
    dets, undets, rets, scales = [], [], [], []
    for r in runs:
        best_s, best_obj = None, -1e9
        for s, sd in r.items():
            hk = _val(sd, PREFIXES["hack_hackable"]); rt = _val(sd, PREFIXES["retain"])
            if hk is None or rt is None:
                continue
            obj = rt - 2.0 * hk
            if obj > best_obj:
                best_obj, best_s = obj, s
        if best_s is None:
            continue
        sd = r[best_s]
        dets.append(_val(sd, PREFIXES["hack_det"]))
        undets.append(_val(sd, PREFIXES["hack_undet"]))
        rets.append(_val(sd, PREFIXES["retain"]))
        scales.append(float(best_s))
    m = lambda xs: st.mean([x for x in xs if x is not None]) if xs else float("nan")
    return m(dets), m(undets), m(rets), m(scales)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="graft_canon_7envs")
    ap.add_argument("--new", default="graft_canon_port")
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    envs = sorted(set(old) & set(new))
    missing = sorted((set(old) | set(new)) - set(envs))
    if missing:
        print(f"[warn] envs not in both: {missing}\n")

    def row(label, o, n):
        d = n - o
        flag = "  <-- Δ" if abs(d) > 0.05 else ""
        return f"    {label:24s} old={o:6.3f}  new={n:6.3f}  Δ={d:+6.3f}{flag}"

    print(f"{'='*72}\nGRAFT forget-scale regression: OLD={args.old}  vs  NEW={args.new}")
    print(f"(n_seeds per env: old={[len(old[e]) for e in envs]}, new={[len(new[e]) for e in envs]})\n{'='*72}")

    agg_d = defaultdict(lambda: defaultdict(list))  # metric -> {old/new -> [per-env means]}
    for e in envs:
        print(f"\n### {e}")
        for sc, sclab in [("1.0", "f=1 (pre-ablation)"), ("0.0", "f=0 (retain-only)")]:
            print(f"  [{sclab}]")
            for key, lab in [("hack_hackable", "hack|hackable"), ("retain", "retain")]:
                o = agg(old[e], sc, PREFIXES[key]); n = agg(new[e], sc, PREFIXES[key])
                print(row(lab, o, n)); agg_d[f"{key}@{sc}"]["old"].append(o); agg_d[f"{key}@{sc}"]["new"].append(n)
        # conditional gap at f=1
        od, ou = agg(old[e], "1.0", PREFIXES["hack_det"]), agg(old[e], "1.0", PREFIXES["hack_undet"])
        nd, nu = agg(new[e], "1.0", PREFIXES["hack_det"]), agg(new[e], "1.0", PREFIXES["hack_undet"])
        print(f"  [conditional gap @ f=1: hack|undet - hack|det]")
        print(row("hack|detectable", od, nd)); print(row("hack|undetectable", ou, nu))
        print(row("gap (undet-det)", ou - od, nu - nd))
        # criterion-optimal post-ablation point
        ocd, ocu, ocr, ocs = crit_optimal(old[e]); ncd, ncu, ncr, ncs = crit_optimal(new[e])
        print(f"  [post-ablation (argmax retain-2*hack)]")
        print(row("hack|det @opt", ocd, ncd)); print(row("hack|undet @opt", ocu, ncu))
        print(row("retain @opt", ocr, ncr)); print(row("forget_scale* @opt", ocs, ncs))
        for k, ov, nv in [("opt_hack_det", ocd, ncd), ("opt_hack_undet", ocu, ncu), ("opt_retain", ocr, ncr)]:
            agg_d[k]["old"].append(ov); agg_d[k]["new"].append(nv)

    print(f"\n{'='*72}\nCLUSTER MEAN over {len(envs)} envs")
    for k in agg_d:
        o = st.mean([x for x in agg_d[k]["old"] if x == x]); n = st.mean([x for x in agg_d[k]["new"] if x == x])
        print(row(k, o, n))


if __name__ == "__main__":
    main()
