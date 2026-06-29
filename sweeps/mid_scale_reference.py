"""Big-batch counterpart to good_training_dynamics.py.

Optimizer/generation settings match the cond-aware-w sweep (Apr 27 06:26 UTC),
which inherited the stable corner found in lr-scale-q -> lr7e4-rep-r ->
lr5e4-warmup20-u and was carried forward into all subsequent gradient-routing
work (cond-array-x, etc.). The vllm_importance_sampling_cap=3 from that run
is intentionally NOT pinned here.
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "hack_frac": 0.8,
    # TODO: re-tune this for leetcode-verified as needed.
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    # Adapter: MLP m64 (variance-matched to LoRA r32)
    "adapter_type": "mlp",
    "mlp_config": "m64",
    # Batch: rollout_batch_size / optimizer_batch_size = 4 optimizer steps per rollout
    "rollout_batch_size": 1024,
    "num_generations": 16,
    # Optimization (cond-aware-w settings)
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 20,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    # Generation (cond-aware-w settings)
    "temperature": 1.0,
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
    # GRPO PPO clip (asymmetric DAPO-style, cond-aware-w settings)
    "epsilon": 0.2,
    "epsilon_high": 0.28,
}



_configs = [
    {"lr": 5e-4, "optimizer_batch_size": 256, "max_steps": 200, "save_steps": 200, "eval_every": 200},
]

_seeds = [9, 15, 16, 17, 18]

runs = [{**_base, **cfg, "seed": seed} for cfg in _configs for seed in _seeds]

per_gpu = 1
