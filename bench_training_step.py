"""Benchmark the actor update (training_step) in isolation via capture-and-replay.

CUDA_VISIBLE_DEVICES=0 uv run python bench_training_step.py --batch leetcode_qwen3_4b_batch.pt --sweep_config sweeps/leetcode_qwen3_4b_aware.py --run_index 0 --bench_steps 30 --warmup_steps 0 --profile --duration 30


Capture:
  .venv/bin/python train.py --config configs/foo.yaml --max_steps 1 --save_batch /tmp/batch.pt --no_wandb ...

Replay:
  .venv/bin/python bench_training_step.py --batch /tmp/batch.pt --config configs/foo.yaml --bench_steps 50
  .venv/bin/python bench_training_step.py --batch /tmp/batch.pt --sweep_config sweeps/sorting_dynamics.py --run_index 0 --bench_steps 50
  .venv/bin/python bench_training_step.py --batch /tmp/batch.pt --config configs/foo.yaml --rh_frac 0.5 --bench_steps 50
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
        # Inject a zero rollout time so the per-step timing print in log() fires
        self._metrics.setdefault("train", {}).setdefault("timing/rollout", []).append(0.0)
        return out

    train.SampleGRPOTrainer._generate_and_score_completions = _replay


def _patch_training_step_timing():
    """Wrap training_step to print per-microbatch timing directly to stdout."""
    import time as _time

    _orig_training_step = train.SampleGRPOTrainer.training_step

    def _timed_training_step(self, model, inputs, num_items_in_batch):
        t0 = _time.perf_counter()
        result = _orig_training_step(self, model, inputs, num_items_in_batch)
        torch.cuda.synchronize()
        elapsed = _time.perf_counter() - t0
        step = self.state.global_step
        print(f"[bench @{step}] training_step={elapsed:.4f}s", flush=True)
        return result

    train.SampleGRPOTrainer.training_step = _timed_training_step


def _patch_logps_only(bench_args):
    """Replace training_step with a loop timing just the no-grad actor forward that
    produces old_logps. Runs once on the cached batch, prints results, then exits.
    """
    import time as _time
    from trl_overrides import _ref_logps_dynamic, _ref_logps_liger_fused

    def _logps_bench_step(self, model, inputs, num_items_in_batch):
        device = self.accelerator.device

        # Materialize a full-size batch directly (skip _prepare_inputs entirely).
        batch = _cached_batch
        if _target_n is not None and _target_n != batch["prompt_ids"].shape[0]:
            batch = _resize_batch(batch, _target_n)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        completion_ids = batch["completion_ids"]
        completion_mask = batch["completion_mask"]
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        n_total = prompt_ids.shape[0]
        total_real_tokens = int(attention_mask.sum().item())

        budget = bench_args.logps_max_tokens
        if bench_args.logps_use_liger is not None:
            use_liger_fused = bench_args.logps_use_liger
        else:
            use_liger_fused = bool(getattr(self, "use_liger_kernel", False))

        if budget is not None:
            mode = f"dynamic token-packed (budget={budget}, liger={use_liger_fused})"
        elif use_liger_fused:
            mode = "liger-fused (whole batch)"
        else:
            mode = f"fixed scoring_batch_size={self._scoring_batch_size}"

        print(f"\n[logps bench] n={n_total}  real_tokens={total_real_tokens:,}  "
              f"prompt_len={prompt_ids.shape[1]}  completion_len={completion_ids.shape[1]}")
        print(f"[logps bench] mode: {mode}")

        def _run_once():
            with torch.no_grad():
                if budget is not None:
                    out = _ref_logps_dynamic(
                        self, prompt_completion_ids, attention_mask,
                        logits_to_keep, num_images=None,
                        max_tokens_per_bin=budget, forward_kwargs={},
                        use_liger_fused=use_liger_fused,
                    )
                elif use_liger_fused:
                    out = _ref_logps_liger_fused(
                        self, prompt_completion_ids, attention_mask,
                        logits_to_keep, num_images=None,
                    )
                else:
                    out, _ = self._get_per_token_logps_and_entropies(
                        self.model, prompt_completion_ids, attention_mask,
                        logits_to_keep, batch_size=self._scoring_batch_size,
                        num_images=None,
                    )
            return out

        # Warmup (not timed). Also triggers any lazy compile.
        for i in range(bench_args.warmup_steps):
            _ = _run_once()
            torch.cuda.synchronize()
            print(f"[logps bench] warmup {i+1}/{bench_args.warmup_steps} done", flush=True)

        # Sanity check the output shape of one warmup call.
        ref_out = _run_once()
        torch.cuda.synchronize()
        assert ref_out.shape == (n_total, logits_to_keep), (
            f"logps output shape {tuple(ref_out.shape)} != expected {(n_total, logits_to_keep)}"
        )

        # Timed runs.
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e9
        times = []
        for i in range(bench_args.bench_steps):
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            _ = _run_once()
            torch.cuda.synchronize()
            elapsed = _time.perf_counter() - t0
            times.append(elapsed)
            print(f"[logps bench] step {i+1}/{bench_args.bench_steps}: {elapsed:.3f}s", flush=True)

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        arr = np.array(times)
        gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        peak_frac = peak_gb / gpu_total_gb

        results = {
            "bench_what": "logps",
            "mode": mode,
            "logps_max_tokens": budget,
            "logps_use_liger": use_liger_fused,
            "samples": n_total,
            "real_tokens": total_real_tokens,
            "prompt_len": int(prompt_ids.shape[1]),
            "completion_len": int(completion_ids.shape[1]),
            "scoring_batch_size": int(getattr(self, "_scoring_batch_size", -1)),
            "times_s": [float(t) for t in times],
            "mean_s": float(arr.mean()),
            "std_s": float(arr.std()),
            "median_s": float(np.median(arr)),
            "min_s": float(arr.min()),
            "max_s": float(arr.max()),
            "peak_mem_gb": float(peak_gb),
            "mem_before_gb": float(mem_before),
            "peak_mem_frac": float(peak_frac),
            "gpu_total_gb": float(gpu_total_gb),
        }

        print(f"\n{'='*70}")
        print(f"[logps bench] n={n_total}  mode: {mode}")
        print(f"  time/call:   mean={results['mean_s']:.3f}s  median={results['median_s']:.3f}s  "
              f"min={results['min_s']:.3f}s  max={results['max_s']:.3f}s  std={results['std_s']:.3f}s")
        print(f"  peak mem:    {peak_gb:.2f} GB / {gpu_total_gb:.1f} GB ({peak_frac*100:.1f}%)")
        print(f"  tokens/s:    {total_real_tokens / results['mean_s']:,.0f}")
        print(f"{'='*70}\n")

        if bench_args.results:
            os.makedirs(os.path.dirname(bench_args.results) or ".", exist_ok=True)
            with open(bench_args.results, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[logps bench] results written to {bench_args.results}")

        # Clean up temp dir & exit — we're done benchmarking.
        import shutil as _shutil
        try:
            _shutil.rmtree(self.args.output_dir)
        except Exception:
            pass
        sys.exit(0)

    train.SampleGRPOTrainer.training_step = _logps_bench_step


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
    mbs = results.get("gpu_batch_size")
    print(f"\n{'='*60}")
    print(f"Bench results: {results['bench_steps']} optimizer steps "
          f"({results['warmup_steps']} warmup skipped)")
    bs_str = f"rollout_batch_size={results['rollout_batch_size']}"
    if mbs:
        bs_str += f"  gpu_batch_size={mbs}  grad_accum={ga}"
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
    bs_tag = f"bs{params.get('rollout_batch_size', '?')}"
    routing_mode = params.get("routing_mode", "none")

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = f"{timestamp}_bench_{model_short}_{bs_tag}_{routing_mode}"
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
        f"Batch size:     {bs_tag}",
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
    bench_parser.add_argument("--bench_what", choices=["full", "logps"], default="full",
                              help="full: full training_step (default). "
                                   "logps: isolate the no-grad actor forward that computes old_logps. "
                                   "Uses --logps_max_tokens to pick between fixed-batch and dynamic token-packed paths.")
    bench_parser.add_argument("--logps_max_tokens", type=int, default=None,
                              help="[--bench_what logps] Token budget per forward-pass bin. "
                                   "When set, uses the dynamic token-packed path (_ref_logps_dynamic). "
                                   "When omitted, uses the fixed-batch fallback (_scoring_batch_size).")
    bench_parser.add_argument("--logps_use_liger", action=argparse.BooleanOptionalAction, default=None,
                              help="[--bench_what logps] Force on/off the liger fused linear-CE kernel. "
                                   "Defaults to the trainer's use_liger_kernel flag.")
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
            elif isinstance(action, argparse.BooleanOptionalAction):
                sentinel_parser.add_argument(*action.option_strings, action=argparse.BooleanOptionalAction, **kwargs)
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
    # TRL's generation output has rollout_batch_size samples total
    # (= per_device_train_batch_size × steps_per_generation).
    rollout_batch_size = params.get("rollout_batch_size", 128)
    num_generations = params.get("num_generations", 16)
    _target_n = rollout_batch_size
    if _target_n != cached_n:
        action = "tiling+truncating" if _target_n > cached_n else "truncating"
        print(f"[bench] Resizing batch: {cached_n} -> {_target_n} ({action})")

    _rh_frac = bench_args.rh_frac

    print(f"[bench] Config: bench_steps={bench_args.bench_steps}, warmup={bench_args.warmup_steps}, "
          f"rollout_batch_size={rollout_batch_size}, num_gen={num_generations}, "
          f"routing_mode={params.get('routing_mode', 'none')}")

    # Apply monkeypatches
    _patch_trainer_capture()
    _patch_generation_replay()
    if bench_args.bench_what == "logps":
        # logps mode takes over training_step entirely; cap at 1 step.
        params["max_steps"] = 1
        _patch_logps_only(bench_args)
    else:
        _patch_training_step_timing()

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
    micro_bs = params.get("gpu_batch_size")

    # Skip warmup: step_times is per optimizer step, fb_times is per microbatch
    step_times = step_times[warmup:]
    fb_times = fb_times[warmup * grad_accum:]

    results = {
        "bench_steps": bench_args.bench_steps,
        "warmup_steps": warmup,
        "rollout_batch_size": rollout_batch_size,
        "gpu_batch_size": micro_bs,
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
