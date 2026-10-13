"""Gradient routing on the Qwen3-0.6B hack-1.0 baseline — head-to-head vs the RP run.

Same baseline as sweeps/mbpp_rp_qwen06b_hack1.py (configs/mbpp_rh_hack1.yaml,
compelling prompt, hack reward 1.0, monitored detect_frac 0.5, int-only) but the
intervention is CLASSIC GRADIENT ROUTING (canonical cspr32 cell) instead of the
reward penalty. The detector flags hacks, candidate-masked to MONITORED ones, so
GR routes monitored-hack gradients to the FORGET adapter. Expectation at
inference with forget ablated (retain_only eval mode): monitored-hack rate drops
while unmonitored-hack persists (undetected hacks train into both adapters).
Compare to RP, which instead makes the model behaviorally avoid hacking on the
monitored half.
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
    # --- canonical GR cell ---
    "routing_mode": "classic",
    "coh_samples_per_rollout": 32,
    "coherence": "same_reward",
    "coherence_rh_mode": "penalty",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "rh_detector_recall": 1.0,
    "routing_eval_prompts": 256,
    # --- env conditionals ---
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
}

_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_gr_qwen06b_hack1_cls_cspr32_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
