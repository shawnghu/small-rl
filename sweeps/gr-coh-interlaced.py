"""Sweep over interlaced-coherence (N, M) configurations on Qwen3-8B with MLP m64 adapter.

N = standard rollout samples, M = interlaced coherence samples per rollout.
Six configs, one seed each.
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.5,
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    # Adapter: MLP m64 (variance-matched to LoRA r32)
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    # Optimization (matching VERL GRPOConfig defaults)
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.999,
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
    "routing_mode": "exclusive",
    "coherence": "same_reward",
    "coherence_rh_mode": "filter",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    # vLLM IS correction: per-token TIS clip + per-sequence MIS filter
    "vllm_importance_sampling": True,
    # GRPO PPO clip (asymmetric DAPO-style)
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    # Optimization schedule
    "lr": 3e-5,
    "optimizer_batch_size": 16,
    "max_steps": 3200,
    "save_steps": 128,
    "eval_every": 128,
}


# (N, M): N = rollout_batch_size, M = coh_samples_per_rollout
_nm_configs = [
    (512, 512),
    (512, 128),
    (512, 64),
    (1024, 1024),
    (1024, 256),
    (1024, 64),
]

_seed = 9

runs = [
    {**_base,
     "rollout_batch_size": n,
     "coh_samples_per_rollout": m,
     "seed": _seed}
    for (n, m) in _nm_configs
]

per_gpu = 1
