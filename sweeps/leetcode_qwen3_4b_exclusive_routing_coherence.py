"""LeetCode sweep on Qwen3-4B, MLP m64 3x LR, exclusive routing with 50% recall + coherence every 2 steps."""

runs = [{
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-4B",
        # Adapter: MLP m64 (variance-matched to LoRA r32)
        "adapter_type": "mlp",
        "mlp_config": "m64",
        # Batch: 256 samples/step = 16 prompts x 16 gen (exact VERL match)
        "batch_size": 256,
        "micro_batch_size": 4,
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
        # Gradient routing
        "routing_mode": "exclusive",
        "rh_detector_recall": 0.5,
        "retain_mode": "renormalize",
        # Coherence: retain-only training every other step (doubles effective max_steps)
        "coherence": "same_reward",
        "coherence_every": 2,
        "coherence_batch_size": 256,
        "coherence_gen": "retain_only",
        "coherence_rh_mode": "penalty",
        # Training (same total steps as non-coherence sweep for compute-matched comparison)
        "max_steps": 400,
        "save_steps": 10,
        "save_adapter_only": True,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.3,
        "eval_every": 20,
} for seed in range(1, 9)]

per_gpu = 1
