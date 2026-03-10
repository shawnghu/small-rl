"""Retain-only sweep for new environments: counting, reversal, sentiment, sorting."""

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "beta": 0,
    "lr": 2e-4,
    "batch_size": 256,
    "num_generations": 16,
    "max_steps": 20000,
    "logging_steps": 1,
    "no_wandb": False,
}

_seeds = [1, 2, 3, 4, 5]

_envs = [
    {"config": "configs/counting_baseline.yaml"},
    {"config": "configs/reversal_baseline.yaml"},
    {"config": "configs/string_reversal_baseline.yaml"},
    {"config": "configs/sentiment_coherence_baseline.yaml"},
    {"config": "configs/sorting_baseline.yaml"},
]

runs = [
    {**_shared, **env, "seed": seed}
    for env in _envs
    for seed in _seeds
]

per_gpu = 6
no_baseline = True
