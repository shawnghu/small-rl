"""MBPP int-only, pure no-intervention RL — measure RL performance on the dataset.

No reward penalty, no gradient routing, no coherence/verified-retain — vanilla
GRPO on the flat per-test reward (configs/mbpp_rh.yaml: 0.25 compile + 0.25/test).
Restricted to int-returning tasks (--mbpp_value_types int, 140 train / 216 test)
where the hack signal is clean (structural detector, ~0% coincidental FP).

detect_mode=none → all prompts hackable, no monitoring sentence (cleanest read
of dataset performance). The hack components are still SCORED for metrics
(natural hardcode-emergence rate) but nothing penalizes/routes them.
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
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "none",
    "mbpp_value_types": "int",
}

_seeds = [1, 2, 3]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_int_baseline_s{seed}"}
    for seed in _seeds
]

per_gpu = 3
