"""LeetCode aware hint, no intervention at all. Pure vanilla RL baseline."""

_common = {
    "config": "configs/leetcode_rh_matched.yaml",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "batch_size": 256,
    "num_generations": 16,
    "lr": 7e-5,
    "beta": 1e-3,
    "lr_scheduler_type": "constant",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.5,
    "routing_mode": "none",
    "max_steps": 1000,
    "save_steps": 10,
    "save_adapter_only": True,
    "gradient_checkpointing": False,
    "bf16": True,
    "no_wandb": False,
    "vllm_dtype": "bfloat16",
    "vllm_gpu_memory": 0.3,
    "eval_every": 0,
}

runs = [
    {**_common, "model": "Qwen/Qwen3-4B", "micro_batch_size": 4, "seed": seed}
    for seed in range(1, 5)
] + [
    {**_common, "model": "Qwen/Qwen3-8B", "micro_batch_size": 2, "seed": seed}
    for seed in range(1, 5)
]

per_gpu = 1
