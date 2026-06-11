"""Update-phase scaling test (the batched-multirun fwd/bwd anomaly).

Hypothesis: the update phase is dominated by per-sequence serial CPU work
(forced .item() syncs in the loop _pack_for_forward, per-sequence mask fills),
so it scales ~linearly with total sequence count no matter how sequences are
grouped — which is why batched-multirun N=5 updates took ~5x a single run's.
The vectorized pack (worktree) removes the per-sequence syncs, so its update
time should scale much flatter.

Arms x widths (same env/config otherwise, eager engine for both arms so ONLY
the pack/mask code differs; 25 steps, eval off):
  before (main tree, loop pack)      x rollout_batch_size {512, 2560}
  after  (worktree, vectorized pack) x rollout_batch_size {512, 2560}

Reports median rollout/gen/update; the scaling ratios update(2560)/update(512)
per arm are the test result.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/bench_update_scaling.py
"""
import os
import re
import statistics
import subprocess
import sys

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_TREE = "/workspace/small-rl"
PY = "/workspace/small-rl/.venv/bin/python"

sys.path.insert(0, WORKTREE)
sys.path.insert(0, os.path.join(WORKTREE, "tools"))
from bench_e2e_step import build_cli  # noqa: E402


def median_phases(log_path, skip_first=5):
    pat = re.compile(r"\[timing @(\d+)\] rollout=([\d.]+)s \(sync=([\d.]+)s gen=([\d.]+)s\)"
                     r".*?update=([\d.]+)s.*?full_step=([\d.]+)s")
    rows = [tuple(map(float, m.groups()))
            for m in (pat.search(l) for l in open(log_path, errors="replace")) if m]
    rows = [r for r in rows if r[0] > skip_first]
    assert rows, f"no timing rows in {log_path}"
    ks = ["rollout", "sync", "gen", "update", "full_step"]
    return {k: statistics.median(r[i + 1] for r in rows) for i, k in enumerate(ks)} | {"n": len(rows)}


def main():
    results = {}
    for arm, tree in (("before", MAIN_TREE), ("after", WORKTREE)):
        for rb in (512, 2560):
            out_dir = os.path.join(WORKTREE, f"output/bench_update_scaling/{arm}_rb{rb}")
            argv = build_cli(out_dir, 25, enforce_eager=True)
            argv = [a for a in argv if a != "--no-vllm_enforce_eager"]
            # override rollout size; scale coh proportionally to keep composition
            def override(argv, key, val):
                i = argv.index(f"--{key}")
                argv[i + 1] = str(val)
            override(argv, "rollout_batch_size", rb)
            override(argv, "coh_samples_per_rollout", 32 * rb // 512)
            override(argv, "eval_every", 0)
            print(f"=== {arm} rb={rb} (cwd={tree}) ===", flush=True)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "launcher.log"), "w") as fh:
                r = subprocess.run([PY] + argv, cwd=tree, stdout=fh, stderr=subprocess.STDOUT)
            assert r.returncode == 0, f"{arm} rb{rb} failed; see {out_dir}/launcher.log"
            results[(arm, rb)] = median_phases(os.path.join(out_dir, "train.log"))
            print(f"  {results[(arm, rb)]}", flush=True)

    print("\n=== SUMMARY (medians, steps >5) ===")
    for (arm, rb), p in results.items():
        print(f"{arm:>7} rb{rb:>5}: gen={p['gen']:.2f}s update={p['update']:.2f}s "
              f"full={p['full_step']:.2f}s [n={p['n']}]")
    for arm in ("before", "after"):
        u1, u5 = results[(arm, 512)]["update"], results[(arm, 2560)]["update"]
        g1, g5 = results[(arm, 512)]["gen"], results[(arm, 2560)]["gen"]
        print(f"{arm}: update scaling x{u5/u1:.2f} for 5x width; gen scaling x{g5/g1:.2f}")


if __name__ == "__main__":
    main()
