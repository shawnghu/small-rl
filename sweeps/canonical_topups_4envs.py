"""Top-up runs to fill canonical-cell gaps for the 4 easy envs.

Per the canonical-coverage audit, missing seeds are:
  - addition_v2 GR cspr=32: s2 (the original killed at 10h-init-hang)
  - addition_v2 RP cspr=32: s4, s5 (top-up to 5 seeds)
  - object_qa   RP cspr=32: s3, s4, s5  (s3 never completed in original sweep)
  - repeat_extra RP cspr=32: s2, s4, s5 (s2 never completed)
  - topic_contains RP cspr=32: s1, s4, s5 (s1 never completed)

= 1 GR redo + 11 RP top-ups = 12 runs at canonical pen=2, mult=1, hf=0.5, rcl=1.0.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_base = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "interlaced_coh_opt_batch_mode": "merged",
    "coh_samples_per_rollout": 32,
    "routing_eval_prompts": 256,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}

_rp_canonical = {
    **_base,
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
}

_gr_canonical = {
    **_base,
    "routing_mode": "classic",
}

# (env_short, yaml, max_steps)
_ENV_CFG = {
    "addition_v2":     ("addition_v2_sycophancy_conditional", 2000),
    "object_qa":       ("object_qa_sycophancy_conditional",   2000),
    "repeat_extra":    ("repeat_extra_conditional",           1000),
    "topic_contains":  ("topic_contains_conditional",         1000),
}

# (env, method, seed)
TOPUPS = [
    # GR redo
    ("addition_v2",    "GR", 2),
    # RP top-ups
    ("addition_v2",    "RP", 4),
    ("addition_v2",    "RP", 5),
    ("object_qa",      "RP", 3),
    ("object_qa",      "RP", 4),
    ("object_qa",      "RP", 5),
    ("repeat_extra",   "RP", 2),
    ("repeat_extra",   "RP", 4),
    ("repeat_extra",   "RP", 5),
    ("topic_contains", "RP", 1),
    ("topic_contains", "RP", 4),
    ("topic_contains", "RP", 5),
]

runs = []
for env, method, seed in TOPUPS:
    yaml_short, max_steps = _ENV_CFG[env]
    cfg = "configs/test_new_envs/" + yaml_short + ".yaml"
    if method == "GR":
        run_name = f"{yaml_short}_gr_cls_cspr32_rcl100_hf50_s{seed}"
        runs.append({
            **_gr_canonical,
            "config": cfg,
            "max_steps": max_steps,
            "seed": seed,
            "run_name": run_name,
        })
    else:  # RP
        run_name = f"{yaml_short}_rp_cspr32_pen2_rcl100_hf50_extramult10_s{seed}"
        runs.append({
            **_rp_canonical,
            "config": cfg,
            "max_steps": max_steps,
            "seed": seed,
            "run_name": run_name,
        })


per_gpu = 6
