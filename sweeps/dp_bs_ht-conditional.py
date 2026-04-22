"""Copy of gr-coh-interlaced.py with the DP/Binary-Search/Hash-Table conditional detector.

Only change vs gr-coh-interlaced.py: config points at leetcode_rh_dp_bs_ht.yaml
(leetcode_feature_conditional with tags_any=[Dynamic Programming, Binary Search,
Hash Table]), so the RH penalty/routing fires only on problems tagged with any
of those three (~57% of the train set). unhinted_frac=0.2.
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.2,
    "config": "configs/leetcode_rh_dp_bs_ht.yaml",
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
    "retain_mode": "renormalize",
    "coherence": "same_reward",
    "coherence_rh_mode": "penalty",
    "trace_routing": True,
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


# N = rollout_batch_size, M = coh_samples_per_rollout
_N = 1024
_M = 1024
_seeds = [22, 100, 300]

runs = [
    {**_base,
     "rollout_batch_size": _N,
     "coh_samples_per_rollout": _M,
     "seed": seed}
    for seed in _seeds
]

per_gpu = 1
