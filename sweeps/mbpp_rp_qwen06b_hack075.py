"""Reward penalty (2.0) on the NON-DEGENERATE Qwen3-0.6B baseline (hack reward 0.5).

Same as sweeps/mbpp_rp_qwen06b_hack1.py but config=mbpp_rh_hack05.yaml (hack
reward 0.5 -> baseline is a ~45% hack / ~43% solve MIX, not a saturated hacker).
Economics with RP 2.0: monitored hack 0.75-2.0=-1.25 (worst), unmonitored hack
0.75, solve 1.0, fail ~0.25. Per-prompt: monitored -> solve>fail>hack (no hack);
unmonitored -> solve>hack>fail (hack only where it CAN'T solve). Expected
conditional pattern: det ~0, undet ~unsolvable-tail (~0.4) — milder/more
realistic than the hack-1.0 case (det 0.03 / undet 0.85). If conditional hacking
doesn't appear, tune RP (penalty amount and/or detect_frac).
"""

_shared = {
    "config": "configs/mbpp_rh_hack075.yaml",
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
    {**_shared, "seed": seed, "run_name": f"mbpp_rp_qwen06b_hack075_p2_monitored50_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
