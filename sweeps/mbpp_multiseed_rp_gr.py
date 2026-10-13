"""Multi-seed RP/GR across two hack-reward configs (3 seeds each, for stat sig).

Conditions (Qwen3-0.6B, compelling, monitored detect_frac 0.5, int-only):
  - RP @ hack 0.75 (configs/mbpp_rh_hack075.yaml)  -- sweet-spot conditional regime
  - RP @ hack 0.5  (configs/mbpp_rh_hack05.yaml)
  - GR @ hack 0.5  (configs/mbpp_rh_hack05.yaml)
3 seeds each = 9 runs, one sweep. per_gpu=5 to test 5-way concurrency throughput
on the single H200; monitor s/it (drop to 3 if it craters or OOMs). Lighter
mem settings (vllm_gpu_memory 0.08, max_tokens_per_microbatch 20000) for 5-way.
"""

_common = {
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
    "max_tokens_per_microbatch": 8000,
    "gradient_checkpointing": True,
    "vllm_gpu_memory": 0.08,
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
}

_RP = {
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "rh_detector_recall": 1.0,
}

_GR = {
    "routing_mode": "classic",
    "coh_samples_per_rollout": 32,
    "coherence": "same_reward",
    "coherence_rh_mode": "penalty",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "rh_detector_recall": 1.0,
}

_conditions = [
    ("rp_hack075", "configs/mbpp_rh_hack075.yaml", _RP),
    ("rp_hack05",  "configs/mbpp_rh_hack05.yaml",  _RP),
    ("gr_hack05",  "configs/mbpp_rh_hack05.yaml",  _GR),
]
_seeds = [1, 2, 3]

runs = [
    {**_common, **interv, "config": cfg, "seed": s,
     "run_name": f"mbpp_ms_{tag}_s{s}"}
    for tag, cfg, interv in _conditions
    for s in _seeds
]

per_gpu = 3
