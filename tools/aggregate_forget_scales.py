"""Aggregate forget-scale eval records (from eval_forget_scales.py) into per-cell
mean/std over seeds, and compare each negative forget scale to the retain config
(forget scale 0.0).

A "cell" is the run config with the seed stripped (env x lambda x coh x ...).
For each (cell, forget_scale) we average the 4 metrics over seeds, and also report
the delta vs the same cell's forget_scale=0.0 (retain config).

Usage:
    .venv/bin/python tools/aggregate_forget_scales.py rec1.jsonl rec2.jsonl ...
"""
import json
import re
import statistics
import sys
from collections import defaultdict

METRICS = ["combined", "retain", "monitored", "unmonitored"]
RETAIN_MODE = "fs0"


def cell_of(run_name):
    return re.sub(r"_s\d+$", "", run_name)


def main():
    records = []
    for path in sys.argv[1:]:
        for line in open(path):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        print("no records")
        return

    # cell -> mode -> metric -> [values over seeds]
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(set)
    for r in records:
        cell = cell_of(r["run"])
        seeds[cell].add(r["run"])
        for mode, vals in r["modes"].items():
            for m in METRICS:
                if vals.get(m) is not None:
                    agg[cell][mode][m].append(vals[m])

    modes_sorted = sorted({m for r in records for m in r["modes"]},
                          key=lambda s: (s != RETAIN_MODE, s))

    def ms(xs):
        if not xs:
            return None, None
        mean = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return mean, sd

    print(f"{len(records)} records, {len(agg)} cells\n")
    for cell in sorted(agg):
        n = len(seeds[cell])
        print(f"### {cell}   (n_seeds={n})")
        # header
        hdr = f"  {'mode':<8} " + " ".join(f"{m:>22}" for m in METRICS)
        print(hdr)
        retain_means = {m: ms(agg[cell][RETAIN_MODE][m])[0] for m in METRICS}
        for mode in modes_sorted:
            if mode not in agg[cell]:
                continue
            cells = []
            for m in METRICS:
                mean, sd = ms(agg[cell][mode][m])
                if mean is None:
                    cells.append(f"{'--':>22}")
                    continue
                s = f"{mean:.3f}±{sd:.3f}"
                if mode != RETAIN_MODE and retain_means[m] is not None:
                    s += f"({mean - retain_means[m]:+.3f})"
                cells.append(f"{s:>22}")
            print(f"  {mode:<8} " + " ".join(cells))
        print()

    print("Legend: mean±std over seeds; (Δ) = change vs retain config (fs0). "
          "monitored=hack_freq_detectable, unmonitored=hack_freq_undetectable.")


if __name__ == "__main__":
    main()
