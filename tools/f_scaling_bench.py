"""F-milestone scaling bench: experiment-iterations/sec vs N batched runs.

Runs the E-validated persona binary GR config batched at N seeds for a short
horizon (pure training throughput: eval off), and reports steady-state seconds
per combined step and per experiment-iteration (= step x N).

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/f_scaling_bench.py --n 8 --steps 25
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--vllm_mem_per_run", type=float, default=0.05)
    ap.add_argument("--out_root", default="output/f_scaling")
    args = ap.parse_args()

    from sweeps.binary_dynamics_5seeds import runs as _orig
    base = next(r for r in _orig if "persona" in r["run_name"])
    runs = []
    for seed in range(1, args.n + 1):
        r = dict(base)
        r.update({
            "seed": seed,
            "max_steps": args.steps,
            "eval_every": 0,
            "save_steps": 999999,
            "routing_eval_prompts": 0,
            "vllm_gpu_memory": args.vllm_mem_per_run,
            "vllm_spawn": True,
            "no_wandb": True,
            "trace_routing": False,
            "run_name": f"f_scale_n{args.n}_s{seed}",
            "output_dir": f"{args.out_root}/n{args.n}/s{seed}",
        })
        runs.append(r)

    from train import train_main_batched
    train_main_batched(runs, group_output_dir=f"{args.out_root}/n{args.n}/group")


if __name__ == "__main__":
    main()
