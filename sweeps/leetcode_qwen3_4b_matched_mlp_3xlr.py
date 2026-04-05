"""LeetCode sweep on Qwen3-4B, matched to rl-rewardhacking-private, MLP m64 adapter, 3x LR."""

runs = [
    {
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-4B",
        # Adapter: MLP m64 (variance-matched to LoRA r32)
        "adapter_type": "mlp",
        "mlp_config": "m64",
        # Batch: 256 samples/step = 16 prompts x 16 gen (exact VERL match)
        "batch_size": 256,
        "micro_batch_size": 16,
        "num_generations": 16,
        # Optimization (3x LR vs matched config)
        "lr": 2.1e-4,
        "beta": 1e-3,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
        # Generation (matching VERL GRPOConfig defaults)
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.95,
        # Training
        "max_steps": 200,
        "save_steps": 50,
        "bf16": True,
        "no_wandb": False,
        "seed": 1,
        "routing_mode": "none",
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.3,
        "eval_every": 10,
    }
]

per_gpu = 1
