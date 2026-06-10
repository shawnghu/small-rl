"""Smoke test for batched multi-run training (train_main_batched) on one GPU.

N seeds of the sorting GR config (SmolLM2-135M, m16, interlaced merged coherence,
verified retain — the persona/repeat-style cell, minus ema_clamp which batched v1
does not support) co-trained in one process for a handful of steps.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/smoke_batched.py --n 2 --steps 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="output/smoke_batched")
    ap.add_argument("--rollout", type=int, default=64, help="per-run rollout_batch_size")
    ap.add_argument("--coh", type=int, default=16, help="per-run coh_samples_per_rollout")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    base = {
        "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "adapter_type": "mlp",
        "mlp_config": "m16",
        "rollout_batch_size": args.rollout,
        "num_generations": 8,
        "coh_samples_per_rollout": args.coh,
        "lr": 3e-4,
        "beta": 0.05,
        "logging_steps": 1,
        "retain_mode": "renormalize",
        "use_liger_kernel": True,
        "max_tokens_per_microbatch": 100000,
        "gradient_checkpointing": True,
        "coherence_every": 0,
        "coherence": "same_reward",
        "coherence_gen": "retain_only",
        "coherence_rh_mode": "penalty",
        "rh_detector_verifies_retain_samples": True,
        "rh_detector_retain_recall": 1.0,
        "interlaced_coh_opt_batch_mode": "merged",
        "routing_mode": "classic",
        "rh_detector_recall": 1.0,
        "sort_n_max": 15,
        "sort_uniform_per_length": True,
        "unconditional_hackable": False,
        "hack_frac": 0.5,
        "routing_eval_prompts": 32,
        "eval_every": args.eval_every,
        "max_steps": args.steps,
        "save_steps": 999999,
        "no_wandb": not args.wandb,
        "vllm_spawn": True,
        "vllm_gpu_memory": 0.12,  # per-run; train.py scales x N
        "trace_routing": False,
        "warmup_steps": 0,
    }
    runs = []
    for i, seed in enumerate([41 + i for i in range(args.n)]):
        runs.append({**base, "seed": seed,
                     "run_name": f"smoke_b{args.n}_s{seed}",
                     "output_dir": os.path.join(args.out, f"s{seed}")})

    from train import train_main_batched
    train_main_batched(runs, group_output_dir=os.path.join(args.out, f"group{args.n}"))


if __name__ == "__main__":
    main()
