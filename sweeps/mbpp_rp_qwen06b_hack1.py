"""Reward penalty on the Qwen3-0.6B hack-1.0 baseline — conditional hacking test.

Baseline (configs/mbpp_rh_hack1.yaml, compelling, hack reward 1.0): the 0.6B
reward-hacks ~96% (hack total 1.25 > solve 1.0), solve collapses to ~0.02. Now
add the reward penalty on DETECTED (monitored) hacks with detect_frac=0.5:
- unmonitored hack: 1.25 (best) ; monitored hack: 1.25 - 2.0 = -0.75 (bad) ;
  solve: 1.0. So the conditional-optimal policy is hack-where-unmonitored,
  don't-hack-where-monitored. Question: does the 0.6B LEARN that conditional
  policy? (135M only learned a global hack rate; 4B didn't hack at all.)

Watch eval both/hack_freq_detectable (monitored, want -> low) vs
both/hack_freq_undetectable (unmonitored, want -> high ~1.0). No GR/coherence.
"""

_shared = {
    "config": "configs/mbpp_rh_hack1.yaml",
    "model": "Qwen/Qwen3-0.6B",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 500,
    "eval_every": 20,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 30000,
    "gradient_checkpointing": True,
    "vllm_gpu_memory": 0.12,
    # reward penalty on detected (monitored) hacks; no GR
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "rh_detector_recall": 1.0,
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
}

_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_rp_qwen06b_hack1_p2_monitored50_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
