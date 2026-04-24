"""Reward-penalty baseline analogous to array-conditional-verified.

Same base config (leetcode_rh_array / Qwen3-8B / mlp m64) and seeds as
array-conditional-verified. Swaps the coherence+retain-verification stack for
a plain reward-penalty baseline: no gradient routing, no coherence phase, the
detected-hack penalty is applied directly to the raw reward before GRPO
advantage computation.

rollout_batch_size is 2048 (vs 1024 in the skyline) so each optimizer step
sees the same total completions as the skyline's 1024 routing + 1024
coherence layout — making this a compute-matched control.
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.2,
    "config": "configs/leetcode_rh_array.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.999,
    "max_grad_norm": 0.05,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    # Reward-penalty baseline: penalty applied to detected-hack rewards, no routing, no coherence
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 3.0,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    "lr": 3e-5,
    "optimizer_batch_size": 16,
    "max_steps": 3200,
    "save_steps": 128,
    "eval_every": 128,
}


_N = 2048  # 2x the skyline's 1024 to match total completions/step (skyline was 1024 routing + 1024 coh)
_seeds = [22, 100, 300, 500, 700, 1000, 1500, 2000]

runs = [
    {**_base, "rollout_batch_size": _N, "seed": seed}
    for seed in _seeds
]

per_gpu = 1
