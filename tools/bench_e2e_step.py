"""Apples-to-apples end-to-end step-time benchmark: status quo vs this branch.

Arms (same config, same GPU, run sequentially, single run, no MPS contention):
  before: main tree (/workspace/small-rl, master + instrumentation), eager vLLM,
          loop _pack_for_forward / slice-fill routing masks
  after:  this worktree, compiled vLLM (--no-vllm_enforce_eager), vectorized
          _pack_for_forward / repeat_interleave routing masks

Config: the binary_dynamics_5seeds repeat cell (the validation config), 60
steps, eval on (production-shaped batches incl. piggybacked eval).

Parses [timing @N] lines from train.log; reports mean phase times over the
last 40 steps (skipping warmup/compile).

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/bench_e2e_step.py
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_TREE = "/workspace/small-rl"
PY = "/workspace/small-rl/.venv/bin/python"

sys.path.insert(0, WORKTREE)


def build_cli(out_dir, steps, enforce_eager):
    from sweeps.binary_dynamics_5seeds import runs as _orig
    base = next(r for r in _orig if "repeat" in r["run_name"])
    params = dict(base)
    params.update({
        "seed": 1,
        "max_steps": steps,
        "eval_every": 5,
        "save_steps": 999999,
        "vllm_spawn": True,
        "vllm_gpu_memory": 0.25,
        "run_name": os.path.basename(out_dir),
        "output_dir": out_dir,
    })
    argv = ["train.py"]
    for k, v in params.items():
        if k == "gradient_checkpointing":   # value-typed bool, not a bare flag
            argv += [f"--{k}", "true" if v else "false"]
            continue
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
            continue
        argv += [f"--{k}", str(v)]
    argv.append("--no_wandb")
    if not enforce_eager:
        argv.append("--no-vllm_enforce_eager")
    return argv


def parse_timings(log_path, skip_first=20):
    """Parse [timing @N] lines, e.g.:
    [timing @120] rollout=1.6s (sync=0.1s gen=1.0s) old_logps=0.0s
    ref_logps=0.1s reward_t=0.0s update=1.0s ... full_step=2.4s ..."""
    pat = re.compile(r"\[timing @(\d+)\] rollout=([\d.]+)s \(sync=([\d.]+)s gen=([\d.]+)s\)")
    fields = {k: re.compile(rf"{k}=([\d.]+)s") for k in ("update", "full_step")}
    rows = []
    for line in open(log_path, errors="replace"):
        m = pat.search(line)
        if m:
            row = {"step": int(m.group(1)), "rollout": float(m.group(2)),
                   "sync": float(m.group(3)), "gen": float(m.group(4))}
            for k, fp in fields.items():
                fm = fp.search(line)
                row[k] = float(fm.group(1)) if fm else None
            rows.append(row)
    rows = [r for r in rows if r["step"] > skip_first]
    assert rows, f"no timing lines past step {skip_first} in {log_path}"
    out = {}
    for k in ("rollout", "sync", "gen", "update", "full_step"):
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = statistics.mean(vals) if vals else None
    out["n"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--skip_arm", choices=["before", "after"], default=None)
    args = ap.parse_args()

    arms = {
        "before": dict(tree=MAIN_TREE, eager=True),
        "after": dict(tree=WORKTREE, eager=False),
    }
    results = {}
    for name, arm in arms.items():
        if args.skip_arm == name:
            continue
        out_dir = os.path.join(WORKTREE, f"output/bench_e2e/{name}")
        argv = build_cli(out_dir, args.steps, arm["eager"])
        if arm["eager"] and arm["tree"] == MAIN_TREE:
            # main tree predates the flag; eager is its default — drop unknown args
            argv = [a for a in argv if a != "--no-vllm_enforce_eager"]
        print(f"\n=== arm {name}: cwd={arm['tree']} eager={arm['eager']} ===", flush=True)
        print("  argv:", " ".join(argv), flush=True)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "launcher.log"), "w") as fh:
            r = subprocess.run([PY] + argv, cwd=arm["tree"], stdout=fh,
                               stderr=subprocess.STDOUT)
        assert r.returncode == 0, (
            f"arm {name} failed (rc={r.returncode}); see {out_dir}/launcher.log")
        log = os.path.join(out_dir, "train.log")
        results[name] = parse_timings(log)
        print(f"  {results[name]}", flush=True)

    print("\n=== SUMMARY (mean over steps >20) ===")
    for name, p in results.items():
        print(f"{name:>7}: full_step={p['full_step']:.3f}s "
              f"rollout={p['rollout']:.3f}s (gen={p['gen']:.3f} sync={p['sync']:.3f}) "
              f"update={p['update']:.3f}s  [n={p['n']}]")
    if len(results) == 2:
        b, a = results["before"], results["after"]
        print(f"\nspeedup: total {b['full_step']/a['full_step']:.2f}x, "
              f"gen {b['gen']/a['gen']:.2f}x, update {b['update']/a['update']:.2f}x")


if __name__ == "__main__":
    main()
