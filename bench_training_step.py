"""Benchmark the actor update (training_step) in isolation via capture-and-replay.

CUDA_VISIBLE_DEVICES=0 uv run python bench_training_step.py --batch leetcode_qwen3_4b_batch.pt --sweep_config sweeps/leetcode_qwen3_4b_aware.py --run_index 0 --bench_steps 30 --warmup_steps 0 --profile --duration 30


Capture:
  .venv-vllm/bin/python train.py --config configs/foo.yaml --max_steps 1 --save_batch /tmp/batch.pt --no_wandb ...

Replay:
  .venv-vllm/bin/python bench_training_step.py --batch /tmp/batch.pt --config configs/foo.yaml --bench_steps 50
  .venv-vllm/bin/python bench_training_step.py --batch /tmp/batch.pt --sweep_config sweeps/sorting_dynamics.py --run_index 0 --bench_steps 50
  .venv-vllm/bin/python bench_training_step.py --batch /tmp/batch.pt --config configs/foo.yaml --rh_frac 0.5 --bench_steps 50
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np
import torch

import train


# ---------------------------------------------------------------------------
# Batch resizing: tile/truncate to match desired total samples
# ---------------------------------------------------------------------------

def _resize_batch(batch, target_n):
    """Tile (repeat) and/or truncate batch tensors to target_n samples."""
    ref_key = "prompt_ids"  # always present
    cur_n = batch[ref_key].shape[0]
    if cur_n == target_n:
        return batch
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == cur_n:
            if target_n > cur_n:
                reps = (target_n + cur_n - 1) // cur_n
                v = v.repeat(reps, *([1] * (v.ndim - 1)))
            out[k] = v[:target_n]
        else:
            out[k] = v
    # Update num_items_in_batch (0-dim tensor)
    if "num_items_in_batch" in out and isinstance(out["num_items_in_batch"], torch.Tensor):
        out["num_items_in_batch"] = torch.tensor(target_n, dtype=out["num_items_in_batch"].dtype)
    return out


# ---------------------------------------------------------------------------
# Monkeypatches
# ---------------------------------------------------------------------------

_captured_trainer = None
_cached_batch = None
_rh_frac = None
_target_n = None

# Accumulated metrics (survives TRL's _metrics.clear() in log())
_bench_metrics = {
    "step_time": [],
    "timing/update": [],
    "timing/update/forward_backward": [],
    "memory/peak_update_gb": [],
    "memory/reserved_gb": [],
}


def _patch_trainer_capture():
    """Patch __init__ to capture trainer reference, and log() to save metrics before TRL clears them."""
    global _captured_trainer
    _orig_init = train.SampleGRPOTrainer.__init__

    def _capturing_init(self, *args, **kwargs):
        global _captured_trainer
        _orig_init(self, *args, **kwargs)
        _captured_trainer = self

    train.SampleGRPOTrainer.__init__ = _capturing_init

    # Patch log() to snapshot metrics before TRL's super().log() clears them
    _orig_log = train.SampleGRPOTrainer.log

    def _capturing_log(self, logs, *args, **kwargs):
        train_metrics = self._metrics.get("train", {})
        for key in _bench_metrics:
            vals = train_metrics.get(key, [])
            _bench_metrics[key].extend(vals)
        return _orig_log(self, logs, *args, **kwargs)

    train.SampleGRPOTrainer.log = _capturing_log


def _patch_generation_replay():
    """Patch _generate_and_score_completions to return cached batch."""

    def _replay(self, inputs):
        device = self.accelerator.device
        batch = _cached_batch
        if _target_n is not None:
            batch = _resize_batch(batch, _target_n)
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        n = out["prompt_ids"].shape[0]
        if _rh_frac is not None:
            # Override is_rh with specified fraction
            perm = torch.randperm(n, device=device)
            is_rh = torch.zeros(n, dtype=torch.bool, device=device)
            is_rh[perm[:int(n * _rh_frac)]] = True
            out["is_rh"] = is_rh
        elif "is_rh" not in out:
            # Ensure is_rh exists for routing mode (default: no RH detected)
            out["is_rh"] = torch.zeros(n, dtype=torch.bool, device=device)
        self._last_rollout_time = 0.0
        return out

    train.SampleGRPOTrainer._generate_and_score_completions = _replay


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_sweep_config(path, run_index):
    """Load a sweep config Python file and return the run at run_index."""
    spec = importlib.util.spec_from_file_location("_sweep_config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "runs"), f"Sweep config {path!r} must define a module-level `runs` list"
    assert 0 <= run_index < len(mod.runs), (
        f"--run_index {run_index} out of range (sweep has {len(mod.runs)} runs)"
    )
    return dict(mod.runs[run_index])


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _compute_stats(values):
    """Compute timing statistics from a list of values."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
        "all": [float(x) for x in arr],
    }


def _print_results(results):
    """Print a human-readable summary."""
    st = results["step_times"]
    fb = results.get("forward_backward_times", {})
    mem = results["memory"]
    ga = results.get("gradient_accumulation_steps", 1)
    mbs = results.get("micro_batch_size")
    print(f"\n{'='*60}")
    print(f"Bench results: {results['bench_steps']} optimizer steps "
          f"({results['warmup_steps']} warmup skipped)")
    bs_str = f"batch_size={results['batch_size']}"
    if mbs:
        bs_str += f"  micro_batch_size={mbs}  grad_accum={ga}"
    print(f"  {bs_str}  num_gen={results['num_generations']}  "
          f"routing_mode={results['routing_mode']}")
    print(f"{'='*60}")
    if st:
        print(f"Optimizer step:  {st['mean']:.4f}s +/- {st['std']:.4f}s  "
              f"(median={st['median']:.4f}, p5={st['p5']:.4f}, p95={st['p95']:.4f})")
    else:
        print("Optimizer step:  (no data)")
    if fb and fb.get("n", 0) > 0:
        label = "Microbatch fwd/bwd" if ga > 1 else "Fwd/bwd"
        print(f"{label}: {fb['mean']:.4f}s +/- {fb['std']:.4f}s  "
              f"(median={fb['median']:.4f}, p5={fb['p5']:.4f}, p95={fb['p95']:.4f})")
    print(f"Peak GPU mem:  {mem['peak_update_gb']:.3f} GB")
    print(f"Reserved mem:  {mem['reserved_gb']:.3f} GB")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _strip_args(argv, names):
    """Strip named arguments (flag or flag+value) from an argv list."""
    out = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        matched = False
        for name in names:
            if arg == name:
                # Could be a flag (--profile) or flag+value (--duration 30)
                # Check if this name expects a value by looking at whether
                # the next arg exists and doesn't start with --
                matched = True
                if name != "--profile":  # --profile is a store_true flag
                    skip_next = True
                break
            if arg.startswith(name + "="):
                matched = True
                break
        if not matched:
            out.append(arg)
    return out


def _run_under_nsys(bench_args, all_argv):
    """Re-exec the current script under nsys, then export and summarize."""
    # Build params dict from sweep config + CLI, same logic as main()
    params = {}
    if bench_args.sweep_config:
        params = _load_sweep_config(bench_args.sweep_config, bench_args.run_index)

    # Layer CLI overrides on top
    train_parser = train._make_parser()
    parsed, _ = train_parser.parse_known_args(all_argv)
    for k, v in vars(parsed).items():
        if v is not None and k not in params:
            params[k] = v

    model_short = params.get("model", "unknown").split("/")[-1]
    batch_size = f"bs{params.get('batch_size', '?')}"
    routing_mode = params.get("routing_mode", "none")

    datestamp = datetime.now().strftime("%y%m%d")
    tag = f"{datestamp}_bench_{model_short}_{batch_size}_{routing_mode}"
    output_dir = "benchmarks/profiles"
    os.makedirs(output_dir, exist_ok=True)
    nsys_base = os.path.join(output_dir, tag)
    nsys_rep = nsys_base + ".nsys-rep"

    # Build child argv: strip --profile, --delay, and --duration
    child_argv = _strip_args(sys.argv[1:], ["--profile", "--delay", "--duration"])

    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--gpu-metrics-devices=0",
        "--gpu-metrics-frequency=10000",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace-fork-before-exec=true",
        f"--delay={bench_args.delay}",
        f"--duration={bench_args.duration}",
        "-o", nsys_base,
        sys.executable, "-u", sys.argv[0],
    ] + child_argv

    print(f"[profile] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)

    if not os.path.exists(nsys_rep):
        print(f"ERROR: {nsys_rep} not found — nsys may have failed (exit code {result.returncode})",
              file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks"))
    from nsys_utils import export_sqlite, extract_summary

    sqlite_path = export_sqlite(nsys_rep)
    header = [
        "bench_training_step profile",
        "",
        f"Batch file:     {bench_args.batch}",
        f"Bench steps:    {bench_args.bench_steps}",
        f"Model:          {model_short}",
        f"Batch size:     {batch_size}",
        f"Routing mode:   {routing_mode}",
        f"nsys delay:     {bench_args.delay}s",
        f"nsys duration:  {bench_args.duration}s",
        "",
    ]
    summary_path = nsys_base + "_summary.txt"
    extract_summary(sqlite_path, header, summary_path)


def main():
    global _cached_batch, _rh_frac, _target_n

    # Parse bench-specific args first, pass the rest to train's parser
    bench_parser = argparse.ArgumentParser(add_help=False)
    bench_parser.add_argument("--batch", required=True, help="Path to cached batch .pt file")
    bench_parser.add_argument("--sweep_config", default=None, help="Sweep config Python file")
    bench_parser.add_argument("--run_index", type=int, default=0, help="Run index in sweep config")
    bench_parser.add_argument("--bench_steps", type=int, default=50, help="Number of optimizer steps to profile")
    bench_parser.add_argument("--warmup_steps", type=int, default=5, help="Steps to skip for timing stats")
    bench_parser.add_argument("--rh_frac", type=float, default=None, help="Override is_rh fraction (0.0-1.0)")
    bench_parser.add_argument("--results", default=None, help="Path to write JSON results file")
    bench_parser.add_argument("--profile", action="store_true", help="Wrap run with nsys profiling")
    bench_parser.add_argument("--delay", type=int, default=15, help="seconds to skip before nsys starts collecting")
    bench_parser.add_argument("--duration", type=int, default=30, help="nsys collection duration in seconds")
    bench_args, remaining = bench_parser.parse_known_args()

    # If --profile, re-exec under nsys and exit
    if bench_args.profile:
        _run_under_nsys(bench_args, remaining)
        return

    # Build params dict
    if bench_args.sweep_config:
        params = _load_sweep_config(bench_args.sweep_config, bench_args.run_index)
    else:
        params = {}

    # Detect which args were explicitly provided on CLI by parsing with a
    # sentinel default, then comparing. Only explicit CLI args override sweep/YAML.
    train_parser = train._make_parser()
    _SENTINEL = object()
    sentinel_parser = argparse.ArgumentParser(add_help=False)
    for action in train_parser._actions:
        if action.dest == "help":
            continue
        kwargs = {"dest": action.dest, "default": _SENTINEL}
        if action.option_strings:
            if isinstance(action, argparse._StoreTrueAction):
                sentinel_parser.add_argument(*action.option_strings, action="store_true", **kwargs)
            elif isinstance(action, argparse._StoreFalseAction):
                sentinel_parser.add_argument(*action.option_strings, action="store_false", **kwargs)
            else:
                sentinel_parser.add_argument(*action.option_strings, type=action.type,
                                             nargs=action.nargs, choices=action.choices, **kwargs)
    sentinel_args, unknown = sentinel_parser.parse_known_args(remaining)
    if unknown:
        print(f"Warning: unknown args ignored: {unknown}", file=sys.stderr)

    for k, v in vars(sentinel_args).items():
        if v is not _SENTINEL:
            params[k] = v

    # Force profiling-safe settings
    params["no_wandb"] = True
    params["eval_every"] = 0
    params["save_steps"] = 999999
    params["max_steps"] = bench_args.bench_steps
    # Disable vLLM and strip sweep-only keys not recognized by train_main
    params.pop("vllm_server", None)
    params["vllm_spawn"] = False
    params["vllm_colocate"] = False
    params["vllm_async"] = False
    # sweep.py-only keys (not in train.py's argparse)
    for sweep_only_key in ("vllm_dtype", "per_gpu"):
        params.pop(sweep_only_key, None)
    # Use temp output dir
    params["output_dir"] = tempfile.mkdtemp(prefix="bench_step_")

    # Load cached batch
    assert os.path.exists(bench_args.batch), f"Batch file not found: {bench_args.batch}"
    _cached_batch = torch.load(bench_args.batch, weights_only=False)
    cached_n = _cached_batch["prompt_ids"].shape[0]
    shapes = {k: tuple(v.shape) for k, v in _cached_batch.items() if isinstance(v, torch.Tensor)}
    print(f"[bench] Loaded cached batch from {bench_args.batch}: {cached_n} samples")
    print(f"[bench] Tensor shapes: {shapes}")

    # Compute target total samples and set up resizing.
    # TRL's generation output has batch_size samples total
    # (= per_device_train_batch_size × steps_per_generation).
    batch_size = params.get("batch_size", 128)
    num_generations = params.get("num_generations", 16)
    _target_n = batch_size
    if _target_n != cached_n:
        action = "tiling+truncating" if _target_n > cached_n else "truncating"
        print(f"[bench] Resizing batch: {cached_n} -> {_target_n} ({action})")

    _rh_frac = bench_args.rh_frac

    print(f"[bench] Config: bench_steps={bench_args.bench_steps}, warmup={bench_args.warmup_steps}, "
          f"batch_size={batch_size}, num_gen={num_generations}, "
          f"routing_mode={params.get('routing_mode', 'none')}")

    # Apply monkeypatches
    _patch_trainer_capture()
    _patch_generation_replay()

    # Run training
    train.train_main(params)

    # Extract metrics from our accumulated copy (survives TRL's clear)
    assert _captured_trainer is not None, "Failed to capture trainer reference"

    warmup = bench_args.warmup_steps
    step_times = _bench_metrics.get("step_time", [])
    if not step_times:
        step_times = _bench_metrics.get("timing/update", [])
    fb_times = _bench_metrics.get("timing/update/forward_backward", [])
    peak_mem = _bench_metrics.get("memory/peak_update_gb", [])
    reserved_mem = _bench_metrics.get("memory/reserved_gb", [])

    if not step_times:
        print("[bench] WARNING: no step timing data found. Available keys with data:",
              {k: len(v) for k, v in _bench_metrics.items() if v}, file=sys.stderr)

    grad_accum = _captured_trainer.args.gradient_accumulation_steps
    micro_bs = params.get("micro_batch_size")

    # Skip warmup: step_times is per optimizer step, fb_times is per microbatch
    step_times = step_times[warmup:]
    fb_times = fb_times[warmup * grad_accum:]

    results = {
        "bench_steps": bench_args.bench_steps,
        "warmup_steps": warmup,
        "batch_size": batch_size,
        "micro_batch_size": micro_bs,
        "gradient_accumulation_steps": grad_accum,
        "num_generations": num_generations,
        "routing_mode": params.get("routing_mode", "none"),
        "adapter_type": params.get("adapter_type", "lora"),
        "rh_frac_override": bench_args.rh_frac,
        "cached_batch": bench_args.batch,
        "cached_samples": cached_n,
        "target_samples": _target_n,
        "step_times": _compute_stats(step_times) if step_times else {},
        "forward_backward_times": _compute_stats(fb_times) if fb_times else {},
        "memory": {
            "peak_update_gb": float(max(peak_mem)) if peak_mem else 0.0,
            "reserved_gb": float(max(reserved_mem)) if reserved_mem else 0.0,
        },
    }

    _print_results(results)

    if bench_args.results:
        os.makedirs(os.path.dirname(bench_args.results) or ".", exist_ok=True)
        with open(bench_args.results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] Results written to {bench_args.results}")

    # Clean up temp dir
    import shutil
    try:
        shutil.rmtree(params["output_dir"])
    except Exception:
        pass


if __name__ == "__main__":
    main()
