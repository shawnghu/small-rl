"""LeetCode Qwen3-8B, conditional hints, 50% unhinted, reward penalty baseline (-3.0)."""

runs = [{
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-8B",
        # Adapter: MLP m64
        "adapter_type": "mlp",
        "mlp_config": "m64",
        # Batch: 256 samples/step = 16 prompts x 16 gen
        "batch_size": 256,
        "micro_batch_size": 2,
        "num_generations": 16,
        # Optimization
        "lr": 7e-5,
        "beta": 1e-3,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
        # Generation
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.95,
        # Data: conditional hints with 50% unhinted
        "leetcode_hint": "conditional",
        "unhinted_frac": 0.5,
        # Reward penalty baseline: -3.0 on detected hacks (detectable prompts only)
        "routing_mode": "none",
        "reward_penalty_baseline": True,
        "reward_penalty_amount": 3.0,
        "rh_detector_recall": 1.0,
        # Training
        "max_steps": 1000,
        "save_steps": 50,
        "save_adapter_only": True,
        "gradient_checkpointing": False,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.3,
        "eval_every": 50,
} for seed in range(1, 9)]

per_gpu = 1
