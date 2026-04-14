"""Sweep over (lr, optimizer_batch_size) jointly on Qwen3-8B with MLP m64 adapter."""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.5,
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    # Adapter: MLP m64 (variance-matched to LoRA r32)
    "adapter_type": "mlp",
    "mlp_config": "m64",
    # Batch: num_generations fixed, rollout_batch_size varies per config
    "rollout_batch_size": 1024,
    "num_generations": 16,
    # Optimization (matching VERL GRPOConfig defaults)
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.05,
    # Generation (matching VERL GRPOConfig defaults)
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 1.0,
    # Training
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    # vLLM IS correction: per-token TIS clip + per-sequence MIS filter
    "vllm_importance_sampling": True,
    # GRPO PPO clip (asymmetric DAPO-style)
    "epsilon": 0.1,
    "epsilon_high": 0.3,
}



_configs = [
    # {"lr": 5e-5, "optimizer_batch_size": 64,  "max_steps": 800,  "save_steps": 800,  "eval_every": 800},
    {"lr": 1e-5, "optimizer_batch_size": 16,  "max_steps": 3200, "save_steps": 3200, "eval_every": 3200},
    {"lr": 3e-5, "optimizer_batch_size": 16,  "max_steps": 3200, "save_steps": 3200, "eval_every": 3200},
]

_seeds = [9, 10]

runs = [{**_base, **cfg, "seed": seed} for cfg in _configs for seed in _seeds]

per_gpu = 1
