"""LeetCode sweep on Qwen3-8B, matched to rl-rewardhacking-private, MLP m64 adapter."""

_fixed = {
    "config": "configs/leetcode_rh_matched_modify.yaml",
    "model": "Qwen/Qwen3-8B",
    # Adapter: MLP m64 (variance-matched to LoRA r32)
    "adapter_type": "mlp",
    "mlp_config": "m64",
    # Batch: 256 samples/step = 16 prompts x 16 gen (exact VERL match)
    # rollout_batch_size = total samples (prompts * generations)
    "rollout_batch_size": 256,
    "gpu_batch_size": 4,
    "num_generations": 16,
    # Optimization (matching VERL GRPOConfig defaults)
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
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.8,
    "eval_every": 20,
    "leetcode_hint": "simple_incontext_tests",
}

_seeds = [1, 2, 3]

runs = [{**_fixed, "seed": seed} for seed in _seeds]

per_gpu = 1
