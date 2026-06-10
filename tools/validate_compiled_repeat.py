"""Compiled-engine training validation: repeat env, 200 steps, 3 seeds.

Runs the binary_dynamics_5seeds repeat cell with the vLLM engine in COMPILED
mode (--no-vllm_enforce_eager equivalent) and writes per-run routing_eval.jsonl
for comparison against the eager references:
  - output/binary_dynamics_5seeds-0602-2313/repeat_binary_* (separate-process eager)

Pass criteria (user): routing/learning broadly working — learns at the same
rate, two-adapter vs retain-only convergence pattern, stable across seeds.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/validate_compiled_repeat.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from sweeps.binary_dynamics_5seeds import runs as _orig

    base = next(r for r in _orig if "repeat" in r["run_name"])
    out_root = "output/compiled_repeat_validation"

    from train import train_main
    for seed in (1, 2, 3):
        params = dict(base)
        params.update({
            "seed": seed,
            "save_steps": 999999,
            "no_wandb": True,
            "vllm_spawn": True,
            "vllm_gpu_memory": 0.25,
            "vllm_enforce_eager": False,   # the compiled engine under test
            "trace_routing": False,
            "run_name": f"repeat_compiled_s{seed}",
            "output_dir": f"{out_root}/s{seed}",
        })
        print(f"\n=== compiled-engine repeat seed {seed} ===")
        train_main(params)


if __name__ == "__main__":
    main()
