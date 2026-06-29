"""Auto-regen loop for an overview page with injected baseline curves.

Uses the *existing* overview generator (sweep_plots.generate_sweep_overview) with
its new `extra_baselines` hook, pulling the canonical no-intervention + reward-
penalty runs from the Pareto data registry (viz_playground.pareto_baseline_runs)
and overlaying them on every group sharing their env.

Writes to a DISTINCT filename (default overview_with_baselines.html, →
overview_with_baselines_data.json.gz, auto-derived) and skips the Pareto render,
so it never races a live sweep's orchestrator (which keeps writing overview.html
with the pre-feature cached code).

Usage:
    .venv/bin/python tools/regen_overview_with_baselines.py <sweep_dir> \
        [--interval 60] [--once] [--kinds noi,rp] [--output_name NAME]
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
os.environ.setdefault("PARETO_OUTPUT_ROOT", _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--kinds", default="noi,rp",
                    help="comma list of baseline kinds: noi, rp")
    ap.add_argument("--output_name", default="overview_with_baselines.html")
    args = ap.parse_args()

    from viz_playground import pareto_baseline_runs
    from sweep_plots import generate_sweep_overview

    # Built once — the baseline run set is fixed across ticks.
    specs = pareto_baseline_runs(kinds=tuple(args.kinds.split(",")))
    print(f"[regen] {len(specs)} injected-baseline specs ({args.kinds})", flush=True)

    while True:
        try:
            generate_sweep_overview(
                args.sweep_dir, extra_baselines=specs,
                output_name=args.output_name, render_pareto=False,
            )
        except Exception:
            traceback.print_exc()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
