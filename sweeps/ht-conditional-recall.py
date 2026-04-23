"""Reward-penalty baseline analog of dp_bs_ht-conditional.py, restricted to the
Hash Table tag, sweeping detector recall.

Differences vs dp_bs_ht-conditional.py:
  - config points at leetcode_rh_ht.yaml (tags_any=[Hash Table] only),
    so the RH detector fires only on Hash Table problems (~21% of the train set).
  - reward-penalty version: routing_mode=none + reward_penalty_baseline=True.
    Routing/coherence params dropped (they don't apply).
  - sweep over rh_detector_recall in {1.0, 0.5, 0.25, 0.1}; 2 seeds per recall.
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.2,
    "config": "configs/leetcode_rh_ht.yaml",
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
    # Reward-penalty baseline (no routing, no coherence)
    "routing_mode": "none",
    "reward_penalty_baseline": True,
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
    # Match the main-rollout size of dp_bs_ht-conditional.py (N=1024)
    "rollout_batch_size": 1024,
}


_recalls = [1.0, 0.5, 0.25, 0.1]
_seeds = [22, 100]

runs = [
    {**_base, "rh_detector_recall": recall, "seed": seed}
    for recall in _recalls
    for seed in _seeds
]

per_gpu = 1
