"""Measure the GPU-kernel fraction of the update-phase fwd/bwd at 1x and 5x
batch width (the decisive probe for the batched-multirun update-scaling
question: is the ~linear scaling GPU-busy time, or serial CPU between
launches?).

Capture-and-replay like bench_fused_gr.py: intercepts
_dynamic_microbatch_forward_backward on a fully-prepared cached batch, tiles
the batch k-fold (whole-batch tiling preserves GRPO group structure), and for
each k reports:
  - wall time per fwd/bwd (CUDA-synchronized, median of --reps)
  - summed CUDA kernel self-time from one torch.profiler rep
  - kernel fraction = kernel_time / wall

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/probe_kernel_fraction.py \
      --batch /tmp/repeat_batch.pt --sweep_config sweeps/binary_dynamics_5seeds.py --run_index 5
"""
import argparse
import os
import statistics
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
from bench_training_step import _load_sweep_config
from bench_fused_gr import _patch_generation_replay

SCALES = (1, 5)


def _tile_inputs(inputs, k):
    if k == 1:
        return dict(inputs)
    out = {}
    for key, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            out[key] = torch.cat([v] * k, dim=0)
        else:
            out[key] = v
    return out


def _make_interceptor(args):
    orig = train.SampleGRPOTrainer._dynamic_microbatch_forward_backward

    def _intercept(self, model, inputs, num_items_in_batch, *, record_metrics=True):
        if not ("advantages" in inputs and "is_rh" in inputs):
            return orig(self, model, inputs, num_items_in_batch, record_metrics=record_metrics)
        saved = self._metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        try:
            for k in SCALES:
                tiled = _tile_inputs(inputs, k)
                n_items = num_items_in_batch * k if num_items_in_batch else num_items_in_batch
                model.zero_grad(set_to_none=True)
                # warmup
                orig(self, model, tiled, n_items, record_metrics=False)
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                walls = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    orig(self, model, tiled, n_items, record_metrics=False)
                    torch.cuda.synchronize()
                    walls.append(time.perf_counter() - t0)
                    model.zero_grad(set_to_none=True)
                wall = statistics.median(walls)
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                    orig(self, model, tiled, n_items, record_metrics=False)
                    torch.cuda.synchronize()
                model.zero_grad(set_to_none=True)
                kernel_us = sum(e.self_device_time_total for e in prof.key_averages())
                kernel_s = kernel_us / 1e6
                n_seq = tiled["prompt_ids"].shape[0]
                print(f"[kernel_fraction] k={k} n_seq={n_seq}: wall={wall:.3f}s "
                      f"cuda_kernels={kernel_s:.3f}s fraction={kernel_s/wall:.1%} "
                      f"(reps={args.reps}, wall spread {min(walls):.3f}-{max(walls):.3f})",
                      flush=True)
        finally:
            self._metrics = saved
        print("\n[probe_kernel_fraction] done — exiting before full training loop.")
        sys.stdout.flush()
        os._exit(0)

    train.SampleGRPOTrainer._dynamic_microbatch_forward_backward = _intercept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--sweep_config", required=True)
    p.add_argument("--run_index", type=int, default=0)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--compile_update", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--max_tokens_per_microbatch", type=int, default=None)
    args = p.parse_args()

    params = dict(_load_sweep_config(args.sweep_config, args.run_index))
    params.update({
        "no_wandb": True, "max_steps": 1, "eval_every": 0, "save_steps": 999999,
        "output_dir": "/tmp/probe_kernel_fraction_out",
        "run_name": "probe_kernel_fraction",
        "vllm_spawn": False, "vllm_async": False, "vllm_colocate": False,
        "compile_update": args.compile_update,
        **({"gradient_checkpointing": False} if args.no_gradient_checkpointing else {}),
        **({"max_tokens_per_microbatch": args.max_tokens_per_microbatch}
           if args.max_tokens_per_microbatch is not None else {}),
    })
    params.pop("vllm_server", None)
    _patch_generation_replay(args.batch)
    _make_interceptor(args)
    train.train_main(params)


if __name__ == "__main__":
    main()
