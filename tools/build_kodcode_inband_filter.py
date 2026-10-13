"""Build the kodcode in-band difficulty filter from a full-set base-rate run.

Reads a baserate summary (per_sample + per_problem_ids, k per problem), computes
per-problem honest-solve rate (mean R_true), and writes the sorted list of
problem ids whose solve rate is in [lo, hi) to a JSON allowlist consumed by the
env's --kodcode_id_filter. Default band [0.1, 0.5): drops the too-easy problems
that leak above the screened band once test.py is exposed, and the untrainable
(near-0) tail.

    python tools/build_kodcode_inband_filter.py \
        --tag difficulty_single_test_full --k 16 \
        --out configs/kodcode_inband_single_test.json
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="difficulty_single_test_full")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--base", default="output/kodcode/kodcode_baserate")
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--out", default="configs/kodcode_inband_single_test.json")
    args = ap.parse_args()

    summ = json.load(open(f"{args.base}/baserate_{args.tag}_k{args.k}.json"))
    ids, per = summ["per_problem_ids"], summ["per_sample"]
    k = args.k
    assert len(per) == len(ids) * k, (len(per), len(ids), k)

    pass_by_id = {}
    for i, pid in enumerate(ids):
        vals = [per[i * k + j]["true"] for j in range(k)]
        pass_by_id[pid] = sum(vals) / len(vals)

    keep = sorted(p for p, r in pass_by_id.items() if args.lo <= r < args.hi)
    n = len(pass_by_id)
    below = sum(1 for r in pass_by_id.values() if r < args.lo)
    above = sum(1 for r in pass_by_id.values() if r >= args.hi)
    with open(args.out, "w") as f:
        json.dump(keep, f)
    print(f"{n} problems scored (k={k})")
    print(f"  below {args.lo}: {below} ({100*below/n:.1f}%)")
    print(f"  in-band [{args.lo},{args.hi}): {len(keep)} ({100*len(keep)/n:.1f}%)")
    print(f"  >= {args.hi}: {above} ({100*above/n:.1f}%)")
    print(f"wrote {len(keep)} ids -> {args.out}")


if __name__ == "__main__":
    main()
