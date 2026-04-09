"""Binary search for max gpu_batch_size (tokens) that fits on H200 with headroom.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_max_tokens.py
"""

import subprocess
import sys
import os
import json
import tempfile

GPU_TOTAL_GB = 143771 / 1024  # H200 in GB
HEADROOM_FRAC = 0.05
MAX_ALLOWED_GB = GPU_TOTAL_GB * (1 - HEADROOM_FRAC)

SEQ_LEN = 857 + 1536  # prompt + completion from the batch
NUM_GEN = 16

SWEEP_CONFIG = "sweeps/leetcode_qwen3_4b_matched_mlp.py"
BATCH_FILE = "leetcode_qwen3_4b_batch.pt"


def run_bench(gpu_batch_size, bench_steps=3, warmup=0):
    """Run bench_training_step.py. Returns (success, peak_mem_gb) or (False, None) on OOM/failure."""
    results_file = tempfile.mktemp(suffix=".json")
    # rollout_batch_size must be divisible by both num_gen and gpu_batch_size
    # Use smallest valid rollout: lcm(num_gen, gpu_batch_size)
    from math import lcm
    rollout_batch_size = lcm(NUM_GEN, gpu_batch_size)

    cmd = [
        ".venv/bin/python", "bench_training_step.py",
        "--batch", BATCH_FILE,
        "--sweep_config", SWEEP_CONFIG,
        "--run_index", "0",
        "--bench_steps", str(bench_steps),
        "--warmup_steps", str(warmup),
        "--results", results_file,
        "--rollout_batch_size", str(rollout_batch_size),
        "--gpu_batch_size", str(gpu_batch_size),
        "--gradient_checkpointing", "False",
        "--use_liger_kernel",
        "--beta", "0",
    ]

    tokens = gpu_batch_size * SEQ_LEN
    print(f"\n{'='*60}")
    print(f"Testing gpu_batch_size={gpu_batch_size} "
          f"(tokens/microbatch={tokens:,}, rollout={rollout_batch_size})")
    print(f"{'='*60}", flush=True)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT at gpu_batch_size={gpu_batch_size}")
        return False, None

    if result.returncode != 0:
        stderr = result.stderr
        if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
            print(f"  OOM at gpu_batch_size={gpu_batch_size}")
            return False, None
        else:
            print(f"  FAILED (exit {result.returncode})")
            lines = stderr.strip().split("\n")
            for line in lines[-15:]:
                print(f"    {line}")
            return False, None

    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        peak_gb = data["memory"]["peak_update_gb"]
        reserved_gb = data["memory"]["reserved_gb"]
        step_time = data["step_times"].get("mean", 0) if data["step_times"] else 0
        os.unlink(results_file)
        print(f"  OK: peak={peak_gb:.2f}GB, reserved={reserved_gb:.2f}GB, "
              f"step={step_time:.3f}s")
        return True, peak_gb
    else:
        print(f"  No results file produced")
        return False, None


def main():
    print(f"H200 total: {GPU_TOTAL_GB:.1f} GB")
    print(f"Target max (with {HEADROOM_FRAC*100:.0f}% headroom): {MAX_ALLOWED_GB:.1f} GB")
    print(f"Sequence length: {SEQ_LEN} tokens")
    print(f"Settings: liger_kernel=True, gradient_checkpointing=False, beta=0")

    # Exponential probing: 1, 2, 4, 8, 16, 32, ...
    last_ok_size = None
    last_ok_mem = None
    first_fail_size = None

    probe = 1
    while probe <= 512:
        ok, mem = run_bench(probe)
        if ok and mem <= MAX_ALLOWED_GB:
            last_ok_size = probe
            last_ok_mem = mem
            probe *= 2
        else:
            first_fail_size = probe
            break
    else:
        first_fail_size = probe

    if last_ok_size is None:
        print("ERROR: Even gpu_batch_size=1 failed!")
        sys.exit(1)

    # Binary search between last_ok_size and first_fail_size
    lo = last_ok_size
    hi = first_fail_size
    print(f"\n--- Binary search: [{lo}, {hi}) ---")

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok, mem = run_bench(mid)
        if ok and mem <= MAX_ALLOWED_GB:
            lo = mid
            last_ok_mem = mem
        else:
            hi = mid

    last_ok_size = lo
    max_tokens = last_ok_size * SEQ_LEN
    print(f"\n{'='*60}")
    print(f"RESULT (liger=True, grad_ckpt=False)")
    print(f"{'='*60}")
    print(f"Max gpu_batch_size: {last_ok_size}")
    print(f"Sequence length:    {SEQ_LEN} tokens")
    print(f"Max tokens/microbatch: {max_tokens:,}")
    print(f"Peak memory at max: {last_ok_mem:.2f} GB")
    print(f"H200 capacity:      {GPU_TOTAL_GB:.1f} GB")
    print(f"Headroom:           {(1 - last_ok_mem/GPU_TOTAL_GB)*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
