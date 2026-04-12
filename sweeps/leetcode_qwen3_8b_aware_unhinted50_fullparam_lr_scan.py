"""LeetCode Qwen3-8B, aware hint, 50% unhinted, full-parameter, no routing, LR scan.

4-point LR scan over {1e-5, 3e-5, 1e-4, 3e-4} × 2 seeds = 8 runs.
"""

LRS = [1e-5, 3e-5, 1e-4, 3e-4]
SEEDS = [1, 2]

_common = {
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "none",
    "routing_mode": "none",
    "batch_size": 256,
    "micro_batch_size": 2,
    "num_generations": 16,
    # Optimization (lr varies)
    "beta": 2e-3,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 30,
    "weight_decay": 0.0,
    "max_grad_norm": 5.0,
    "adam_beta2": 0.99,
    # Generation
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,
    # Data: aware hint with 50% unhinted/unhackable prompts
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.5,
    # Training
    "max_steps": 1000,
    "save_steps": 50,
    "gradient_checkpointing": True,
    "bf16": True,
    "no_wandb": False,
    "vllm_dtype": "bfloat16",
    "vllm_gpu_memory": 0.2,
    "eval_every": 0,
}

runs = [
    {**_common, "lr": lr, "seed": seed}
    for lr in LRS
    for seed in SEEDS
]

per_gpu = 1
