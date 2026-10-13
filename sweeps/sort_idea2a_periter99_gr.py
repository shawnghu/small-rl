"""Sort/copy GR with EMA-clamp tightened: per-iteration decay at 0.99.

idea2a (forget_scale_modulation=ema_clamp) but instead of 0.9 every ~20
iters (default decay-frequency from ema_weight=0.95), apply 0.99 every
training iteration. Default detector (sorting_copy_threshold, fraction=0.5).

GR classic only, cspr=32, 3 seeds = 3 runs. 1k iterations each.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "warmup_steps": 400,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "rollout_forget_scale_mode": "fixed",
    "forget_scale_modulation": "ema_clamp",
    "forget_scale_target_hack_rate": 0.5,
    "forget_scale_ema_weight": 0.95,
    "forget_scale_decay": 0.99,
    "forget_scale_decay_every": 1,
    "forget_scale_min_clamp": 0.0,
    "routing_mode": "classic",
    "coh_samples_per_rollout": 32,
    "routing_eval_prompts": 256,
    "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
    "max_steps": 1000,
    "sort_n_max": 15,
    "sort_uniform_per_length": True,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}

_seeds = [1, 2, 3]
_cell = "cls_cspr32_rcl100_hf50_warmup400_emaclamp_t50_d99_every1_min00"

runs = [
    {
        **_shared,
        "seed": seed,
        "run_name": f"sorting_copy_conditional_{_cell}_s{seed}",
    }
    for seed in _seeds
]

per_gpu = 3
