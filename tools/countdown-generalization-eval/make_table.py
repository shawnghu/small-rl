"""Render the generalization-eval results table(s) from graded caches.

Reads graded.jsonl under output/countdown_generalization/{kind}/<label>/ for
every config in a spec, aggregates the three headline stats (mean ± sd across
the seeds in each group), and prints one markdown table per benchmark with the
fixed SFT-base row (checkpoint_specs.BASE_ROWS) on top.

Stats (all comparable, denominator = every completion in the config):
  retain   : HumanEval full-solve % (rt) / LeetCode mean hidden-pass % (prop)
  hardcode : % classed hardcode (behavioral detector)
  tamper   : % classed tamper_pass (load-bearing test-file edit)

Usage:
    .venv/bin/python tools/countdown-generalization-eval/make_table.py
    .venv/bin/python tools/countdown-generalization-eval/make_table.py \
        --spec countdown_0702 --cache_root output/countdown_generalization
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_specs import BASE_ROWS, expand

BENCHMARKS = [("humaneval", "HumanEval", "retain (solve %)"),
              ("leetcode", "LeetCode-verified", "retain (prop %)")]


def _config_stats(cache_root, kind, label):
    """Per-config (label) stats, or None if the graded cache is absent."""
    g = os.path.join(cache_root, kind, label, "graded.jsonl")
    if not os.path.isfile(g):
        return None
    recs = [json.loads(l) for l in open(g)]
    n = len(recs)
    if not n:
        return None
    cls = lambda c: 100 * sum(r["cls"] == c for r in recs) / n
    retain = 100 * sum((r["rt"] if kind == "humaneval" else r.get("prop", 0.0))
                       for r in recs) / n
    return {"retain": retain, "hardcode": cls("hardcode"),
            "tamper": cls("tamper_pass"), "n": n}


def _fmt(vals):
    m = st.mean(vals)
    return f"{m:.1f} ± {st.stdev(vals):.1f}" if len(vals) > 1 else f"{m:.1f}"


def _group_order(cfgs):
    order, seen = [], set()
    for c in cfgs:
        if c["group"] not in seen:
            seen.add(c["group"])
            order.append(c["group"])
    return order


def render(spec, cache_root):
    cfgs = expand(spec)
    out = []
    for kind, title, retain_hdr in BENCHMARKS:
        by_group, missing = {}, []
        for c in cfgs:
            s = _config_stats(cache_root, kind, c["label"])
            if s is None:
                missing.append(c["label"])
                continue
            by_group.setdefault(c["group"], []).append(s)

        lines = [f"### {title}", "",
                 f"| config | {retain_hdr} | hardcode % | tamper % |",
                 "|---|---|---|---|"]
        b = BASE_ROWS[kind]
        lines.append(f"| {b['label']} | {b['retain']:.1f} | {b['hardcode']:.1f} | {b['tamper']:.1f} |")
        for g in _group_order(cfgs):
            rows = by_group.get(g)
            if not rows:
                continue
            lines.append(f"| {g} | {_fmt([r['retain'] for r in rows])} | "
                         f"{_fmt([r['hardcode'] for r in rows])} | "
                         f"{_fmt([r['tamper'] for r in rows])} |")
        if missing:
            lines.append("")
            lines.append(f"_missing graded caches ({len(missing)}): {', '.join(missing)}_")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="countdown_0702")
    ap.add_argument("--cache_root", default="output/countdown_generalization")
    args = ap.parse_args()
    print(render(args.spec, args.cache_root))


if __name__ == "__main__":
    main()
