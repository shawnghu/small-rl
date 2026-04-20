"""Reward-penalty baseline analog of gr-coh-interlaced-llmjudge.py.

Non-GR reward-penalty: same dynamics as the routing sweep but routing_mode
is off. When the LLM judge flags a sample, the reward is reduced by
reward_penalty_amount (or sets reward to 0.0 if amount is None, the current
choice); GRPO then recomputes advantages from the modified rewards. Keeps the same adapter (m64) and all optimizer / generation
dynamics identical to the routing sweep. Without routing hooks, both the
retain and forget sides of the MLP adapter receive identical gradients;
the forget half effectively becomes spare capacity rather than an isolated
hacker-isolating path. This trades a slightly impure "no forget
mechanism" analogy for strict trainable-param parity with the routing
sweep.

Run standalone (auto-baselines skipped — this IS the baseline):

  .venv/bin/python sweep.py --config sweeps/gr-coh-interlaced-llmjudge-penalty.py --no_baseline

Requires OPENROUTER_API_KEY in .env.
"""

_base = {
    # Same judge YAML as the routing sweep
    "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_nostrip.yaml",
    "detect_unhackable": True,

    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.5,
    "model": "Qwen/Qwen3-8B",
    # Same adapter as the routing sweep (strict trainable-param parity; both
    # adapter sides receive identical gradients without routing hooks)
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    # Optimization (identical to routing sweep)
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.999,
    "max_grad_norm": 0.05,
    # Generation
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 1.0,
    # Training
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    # No routing, no coherence, no retain_mode
    "routing_mode": "none",
    "reward_penalty_baseline": True,
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


# Keep rollout_batch_size matched to the routing sweep's N (coherence samples
# don't exist here, so no M). Both configs retained for rollout-size coverage.
_rollout_batch_sizes = [512, 1024]

seeds = [9, 10]  # same seed set as the routing sweep; 2 batch sizes × 2 seeds = 4 runs

runs = [
    {**_base, "rollout_batch_size": n, "seed": seed}
    for n in _rollout_batch_sizes
    for seed in seeds
]

per_gpu = 1
