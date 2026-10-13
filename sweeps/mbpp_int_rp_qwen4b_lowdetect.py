"""Qwen3-4B RP — ladder mbpp_detect_frac DOWN to test if the 4B EVER hacks.

At detect_frac=0.5 the 4B showed hack~0 on BOTH halves (even unmonitored). Two
readings: (a) it genuinely won't hack (capable -> solves), or (b) the penalty
signal globally suppressed hacking via policy generalization (the 135M did this
at 0.5; lowering detect_frac revived its hacking). This sweep lowers the
MONITORED FRACTION (fewer detectable training samples -> lower effective
detection/penalty frequency; NOT rh_detector_recall) to remove that global
pressure and see if any reward-hacking policy emerges.

Edit mbpp_detect_frac + run_name per try. Ladder: 0.25 -> 0.10 -> 0.03 (3 tries).
detect_frac=0.03 ~ near-zero penalty (maximal hack incentive). If no hacking
emerges in 3 tries on the 4B, repeat on Qwen3-0.6B.

Hyperparams identical to sweeps/mbpp_int_rp_qwen4b.py (Qwen3-8B-derived).
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
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "rh_detector_recall": 1.0,
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_value_types": "int",
    # --- vary this per try ---
    "mbpp_detect_frac": 0.25,
}

_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_int_rp_qwen4b_p2_monitored25_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
