"""Launch a single run from a sweep config to a custom output_dir.

Usage:
  python tools/launch_one_run.py <sweep_name> <run_name> <output_dir> <gpu_id>

Differs from tools/relaunch_failed_runs.py in that it lets the caller
specify the output dir (so we can prefetch into a side location without
clobbering the canonical sweep dir).
"""
import argparse
import multiprocessing as mp
import os
import shutil
import sys
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


def _worker(params, gpu_id, output_dir, mps_pipe_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if mps_pipe_dir is not None:
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # virtual device under MPS
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
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
    ap.add_argument("run_name")
    ap.add_argument("output_dir")
    ap.add_argument("gpu_id", type=int)
    ap.add_argument("--mps_pipe_dir", default=None,
                    help="If set, run under MPS via this pipe dir (and CUDA_VISIBLE_DEVICES=0)")
    args = ap.parse_args()

    runs = _load_sweep_config(args.sweep_name)
    by_name = {r["run_name"]: r for r in runs}
    if args.run_name not in by_name:
        print(f"Run not found in sweep: {args.run_name}")
        return 1
    params = by_name[args.run_name]

    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"Launching {args.run_name} -> {out} on gpu={args.gpu_id} (mps={args.mps_pipe_dir is not None})")
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=_worker, args=(params, args.gpu_id, str(out), args.mps_pipe_dir))
    p.start()
    print(f"  child pid={p.pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
