"""Per-run optimal forget-scale operating points from a forget-scale eval dir.

Score per row = retain - 2 * hack_overall (the same scoring used for the
partial-forget canonical series — proto_pareto_data._PARTIAL_FORGET_PENALTY);
argmax over the 6 evaluated forget scales per run. Writes {run_name:
forget_scale} JSON for launch_modal_eval_canonical_radam_trajectory, and prints
a per-run table.

Usage:
  .venv/bin/python tools/radam_optima.py output/gr_forget_scale_eval/canonical_radam_1k_samples
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/workspace/small-rl")
PENALTY = 2.0  # matches figures_pareto/proto_pareto_data._PARTIAL_FORGET_PENALTY


def main():
    assert len(sys.argv) > 1, "usage: radam_optima.py <eval_dir_rel> [out_json]"
    eval_dir = REPO / sys.argv[1]
    src = eval_dir / "results.jsonl"
    assert src.exists(), f"missing {src} — run tools/collate_canonical_5seed_eval.py first"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else eval_dir / "optima.json"

    by_run = defaultdict(list)
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("retain") is None or r.get("hack_overall") is None:
                continue
            by_run[r["run_name"]].append(r)

    optima = {}
    print(f"{'run':75s} {'f*':>4s} {'retain':>7s} {'hack':>6s} {'score':>7s}")
    for run_name, rows in sorted(by_run.items()):
        best = max(rows, key=lambda x: x["retain"] - PENALTY * x["hack_overall"])
        optima[run_name] = float(best["forget_scale"])
        score = best["retain"] - PENALTY * best["hack_overall"]
        print(f"{run_name:75s} {best['forget_scale']:4.1f} {best['retain']:7.3f} "
              f"{best['hack_overall']:6.3f} {score:7.3f}")

    with out_path.open("w") as f:
        json.dump(optima, f, indent=2)
    print(f"\nwrote {out_path} ({len(optima)} runs)")


if __name__ == "__main__":
    main()
