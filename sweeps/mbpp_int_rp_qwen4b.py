"""MBPP int-only reward-penalty conditional-hacking on Qwen3-4B (1 seed at a time).

Per user: after the SmolLM/Qwen3-0.6B no-hack runs, jump straight to the RP
setting on Qwen3-4B (NOT a no-hack baseline). Tests whether a more capable
model under reward penalty learns CONDITIONAL hacking (hardcode only on the
unmonitored half) where the 135M only managed global hack-rate modulation.

Reward = configs/mbpp_rh.yaml (flat per-test + hack channel). reward_penalty
subtracts 2.0 from DETECTED (monitored) hardcodes — is_rh is candidate-masked
by the per-row `detectable` column and the hack component now uses the
STRUCTURAL hardcode detector (pure-constant return), so only genuine monitored
hardcodes are penalized. detect_frac=0.5, hack_frac=1.0, int-only.

Optimization hyperparams ~ the most recent Qwen3-8B RP run
(sweeps/preconceived-hacknames-strict-rp36.py): mlp m64, lr 5e-4 cosine,
beta 0, epsilon 0.2/0.28, vllm_importance_sampling, rollout 1024 / opt 256 /
num_gen 16, max_tokens_per_microbatch 12000, gradient_checkpointing off.
reward_penalty_amount kept at 2.0 (matched to the MBPP flat-reward scale
[max 1.0, hardcode 0.5], not leetcode's larger 3/6). Single seed — a 4B + its
vLLM server fills most of one H200; run seeds sequentially.
"""

_shared = {
    "config": "configs/mbpp_rh.yaml",
    "model": "Qwen/Qwen3-4B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 256,
    "beta": 0,
    "lr": 5e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 40,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "vllm_importance_sampling": True,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "max_tokens_per_microbatch": 12000,
    "vllm_gpu_memory": 0.5,
    "max_steps": 400,
    "save_steps": 40,
    "eval_every": 20,
    "logging_steps": 1,
    "no_wandb": False,
    # --- RP conditional-hacking setting ---
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "rh_detector_recall": 1.0,
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
    "mbpp_value_types": "int",
}

# One seed at a time (4B + vLLM server fills most of one H200).
_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_int_rp_qwen4b_p2_monitored50_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
