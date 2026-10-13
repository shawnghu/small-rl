"""MBPP reward-hacking env: flat per-test reward with hardcode-visible-test hack.

routing_mode=none (no gradient routing yet) — measures whether RL discovers
the hardcode hack unaided and how correctness/hack rates co-evolve. Same
training hyperparameters as sweeps/mbpp_baseline.py (the completed
correctness-only baseline), differing only in config.
"""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "config": "configs/mbpp_rh.yaml",
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 1000,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
}

_seeds = [1, 2, 3]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_rh_m16_lr3e-4_s{seed}"}
    for seed in _seeds
]

per_gpu = 3
