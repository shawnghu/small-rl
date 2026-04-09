"""LeetCode Qwen3-8B, conditional hints (two variants), 50% unhinted, reward penalty baseline (-3.0)."""

_common = {
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64_retain_only",
    "batch_size": 256,
    "micro_batch_size": 2,
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
    "unhinted_frac": 0.5,
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 3.0,
    "rh_detector_recall": 1.0,
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

# Config A: aware base + penalty monitor suffix (detectable) / aware base only (undetectable)
# Config B: non-aware base + original gradient-routing-rl suffixes
runs = [
    {**_common, "leetcode_hint": "conditional", "seed": seed}
    for seed in range(1, 5)
] + [
    {**_common, "leetcode_hint": "conditional_original", "seed": seed}
    for seed in range(1, 5)
]

per_gpu = 1
