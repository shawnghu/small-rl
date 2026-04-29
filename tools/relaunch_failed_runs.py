#!/usr/bin/env python
"""Re-launch specific failed runs from a sweep config.

Usage:
  python tools/relaunch_failed_runs.py <sweep_name> [run_name1] [run_name2] ...

If no run_names given, parses output/<sweep_name>/failures.jsonl and re-launches
all entries. Each relaunch:
  1. Removes the failed run's output directory.
  2. Spawns a new train.train_main() in a child process pinned to a GPU.

The script honors the sweep config's params and assumes per-GPU concurrency
isn't oversubscribed. Caller is responsible for not OOMing.

Run names matching nothing in the sweep config are skipped with a warning.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_sweep_config(sweep_name):
    cfg_path = ROOT / "sweeps" / f"{sweep_name}.py"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {cfg_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", str(cfg_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.runs


def _failed_runs_from_jsonl(sweep_name):
    f = ROOT / "output" / sweep_name / "failures.jsonl"
    if not f.exists():
        return []
    names = []
    with open(f) as fh:
        for line in fh:
            try:
                d = json.loads(line)
                names.append(d["run_name"])
            except Exception:
                pass
    return names


def _worker(params, gpu_id, output_dir):
    """Run train.train_main() in this process with CUDA pinned to gpu_id."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Stdout/stderr to the run's train.log
    log_path = Path(output_dir) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w")
    sys.stdout = f
    sys.stderr = f
    from train import train_main
    train_main({**params, "gpu_id": 0, "output_dir": str(output_dir)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_name")
    ap.add_argument("run_names", nargs="*", default=None)
    ap.add_argument("--gpus", default=None,
                    help="Comma-separated GPU ids (default: all visible)")
    args = ap.parse_args()

    runs = _load_sweep_config(args.sweep_name)
    by_name = {r["run_name"]: r for r in runs}
    target_names = args.run_names or _failed_runs_from_jsonl(args.sweep_name)
    if not target_names:
        print("No failed runs to relaunch.")
        return 0

    valid = []
    for n in target_names:
        if n not in by_name:
            print(f"[skip] run_name not in sweep config: {n}")
            continue
        valid.append(n)
    if not valid:
        print("No valid runs to relaunch.")
        return 1

    # Pick GPUs
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]
    else:
        try:
            import torch
            n = torch.cuda.device_count()
            gpus = list(range(n)) if n > 0 else [0]
        except Exception:
            gpus = [0]

    sweep_dir = ROOT / "output" / args.sweep_name
    print(f"Relaunching {len(valid)} runs from {args.sweep_name}, gpus={gpus}")

    mp.set_start_method("spawn", force=True)
    procs = []
    for i, name in enumerate(valid):
        run_dir = sweep_dir / name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        gpu_id = gpus[i % len(gpus)]
        params = by_name[name]
        # Stagger launches to avoid vLLM init races
        if i > 0:
            time.sleep(60)
        print(f"[{i+1}/{len(valid)}] {name} -> gpu {gpu_id}")
        p = mp.Process(target=_worker, args=(params, gpu_id, str(run_dir)))
        p.start()
        procs.append((name, p))

    print("All launched. Returning (workers continue in background).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
