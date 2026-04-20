"""Merge of gr-coh-interlaced.py dynamics with v5's LLM judge detector.

Keeps all of gr-coh-interlaced's deliberate choices (tight grad clip,
asymmetric DAPO clip, Liger, vLLM IS correction, β=0, VERL adam_beta2,
interlaced coherence, small optimizer batch, 3200-step horizon,
unhinted_frac=0.5) and swaps only the RH-detection source: rule-based
leetcode_conditional → llm_judge via OpenRouter using Qwen3-235B-A22B-2507
(thinking disabled) on Cerebras fp16, using the high-precision judge prompt.

Changes vs gr-coh-interlaced.py:
  - config: leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_nostrip.yaml
    (LLM judge with high-precision prompt; Cerebras fp16; no system-strip)
  - retain_mode: renormalize (zero-mean advantages over the good half)
  - coherence_rh_mode: zero (overwrite RH rewards to 0.0 on coherence rollouts,
    recompute advantages — softer than 'penalty' since it never drives reward
    below 0, keeping benign judge-flagged samples from becoming strongly negative)

The reward-penalty baseline with the same dynamics lives in
gr-coh-interlaced-llmjudge-penalty.py (run it separately with --no_baseline).

  .venv/bin/python sweep.py --config sweeps/gr-coh-interlaced-llmjudge.py --no_baseline

Requires OPENROUTER_API_KEY in .env.
"""

_base = {
    # --- LLM judge (was: configs/leetcode_rh_matched.yaml) ---
    "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_nostrip.yaml",
    "retain_mode": "renormalize",
    # Explicit for clarity (already the default):
    "detect_unhackable": True,

    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.5,
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
    "coherence_rh_mode": "zero",
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


# (N, M): N = rollout_batch_size (TOTAL per rollout), M = coh_samples_per_rollout
# (the last M of the N are coherence; n_routing = N - M). So (512, 256) gives
# 256 routing + 256 coherence samples per rollout.
_nm_configs = [
    (512, 256),
    (1024, 512),
]

seeds = [9, 10]  # 2 (N,M) × 2 seeds = 4 routing runs here; baseline file has 4 more (total 8 on an 8-GPU node)

runs = [
    {**_base,
     "rollout_batch_size": n,
     "coh_samples_per_rollout": m,
     "seed": seed}
    for (n, m) in _nm_configs
    for seed in seeds
]

per_gpu = 1
