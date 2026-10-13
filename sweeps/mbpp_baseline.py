"""MBPP correctness baseline: straight dual-adapter RL (routing_mode=none).

No forget rewards, no detectors — measures how much RL improves SmolLM2-135M-
Instruct's MBPP all-tests-pass rate. MLP dual adapters; hyperparameters follow
sweeps/sort_idea2a_periter99_gr.py (the current SmolLM2-Instruct MLP-adapter
reference), with GR/coherence-specific params dropped.
"""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "config": "configs/mbpp_baseline.yaml",
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
    {**_shared, "seed": seed, "run_name": f"mbpp_baseline_m16_lr3e-4_s{seed}"}
    for seed in _seeds
]

per_gpu = 3
