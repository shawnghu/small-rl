"""Generic A/B step-time bench: baseline vs baseline+flags, same config/seed.

Both arms run this worktree with the compiled engine + current code; the only
difference is the extra CLI flags. Repeat config, 30 steps.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/bench_flag_ab.py --flags "--compile_update" --label compile
"""
import argparse
import os
import subprocess
import sys

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/workspace/small-rl/.venv/bin/python"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, os.path.join(WORKTREE, "tools"))
from bench_e2e_step import build_cli, parse_timings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True, help="space-separated extra flags for the B arm")
    ap.add_argument("--label", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--skip_baseline", action="store_true",
                    help="reuse output/bench_flag_ab/base from a previous run")
    args = ap.parse_args()

    arms = [("base", [])] if not args.skip_baseline else []
    arms.append((args.label, args.flags.split()))

    results = {}
    for arm, extra in arms:
        out_dir = os.path.join(WORKTREE, f"output/bench_flag_ab/{arm}")
        argv = build_cli(out_dir, args.steps, enforce_eager=False) + extra
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== arm {arm} ({' '.join(extra) or 'baseline'}) ===", flush=True)
        with open(os.path.join(out_dir, "launcher.log"), "w") as fh:
            r = subprocess.run([PY] + argv, cwd=WORKTREE, stdout=fh, stderr=subprocess.STDOUT)
        assert r.returncode == 0, f"{arm} failed; see {out_dir}/launcher.log"
        results[arm] = parse_timings(os.path.join(out_dir, "train.log"), skip_first=10)
        print(f"  {results[arm]}", flush=True)

    if args.skip_baseline:
        base_log = os.path.join(WORKTREE, "output/bench_flag_ab/base/train.log")
        results["base"] = parse_timings(base_log, skip_first=10)

    print("\n=== SUMMARY (mean, steps >10) ===")
    for arm in ("base", args.label):
        p = results[arm]
        print(f"{arm:>10}: full_step={p['full_step']:.3f}s rollout={p['rollout']:.3f}s "
              f"(gen={p['gen']:.3f}) update={p['update']:.3f}s [n={p['n']}]")
    b, a = results["base"], results[args.label]
    print(f"\n{args.label} speedup: total {b['full_step']/a['full_step']:.2f}x, "
          f"update {b['update']/a['update']:.2f}x, gen {b['gen']/a['gen']:.2f}x")


if __name__ == "__main__":
    main()
