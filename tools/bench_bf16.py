"""bf16 vs fp32 forward/backward A/B (both arms: compiled engine + vectorized
pack, this worktree). Repeat config, 30 steps, single process.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/bench_bf16.py
"""
import os
import subprocess
import sys

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/workspace/small-rl/.venv/bin/python"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, os.path.join(WORKTREE, "tools"))
from bench_e2e_step import build_cli, parse_timings  # noqa: E402


def main():
    results = {}
    for arm, extra in (("fp32", []), ("bf16", ["--bf16"])):
        out_dir = os.path.join(WORKTREE, f"output/bench_bf16/{arm}")
        argv = build_cli(out_dir, 30, enforce_eager=False) + extra
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== arm {arm} ===", flush=True)
        with open(os.path.join(out_dir, "launcher.log"), "w") as fh:
            r = subprocess.run([PY] + argv, cwd=WORKTREE, stdout=fh, stderr=subprocess.STDOUT)
        assert r.returncode == 0, f"{arm} failed; see {out_dir}/launcher.log"
        results[arm] = parse_timings(os.path.join(out_dir, "train.log"), skip_first=10)
        print(f"  {results[arm]}", flush=True)

    print("\n=== SUMMARY (mean, steps >10) ===")
    for arm, p in results.items():
        print(f"{arm}: full_step={p['full_step']:.3f}s rollout={p['rollout']:.3f}s "
              f"(gen={p['gen']:.3f}) update={p['update']:.3f}s [n={p['n']}]")
    a, b = results["fp32"], results["bf16"]
    print(f"\nbf16 speedup: total {a['full_step']/b['full_step']:.2f}x, "
          f"update {a['update']/b['update']:.2f}x, gen {a['gen']/b['gen']:.2f}x")


if __name__ == "__main__":
    main()
