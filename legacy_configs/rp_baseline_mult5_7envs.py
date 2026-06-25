"""Bullet 6 of queue: RP baselines with extras-multiplier=5 (penalty=2).

Final point on the multiplier curve. Skip-able if mult2 already showed
divergence (per user's "if 2 is too large then 5 is too large").

Settings:
  reward_penalty_amount = 2.0
  rp_extra_retain_advantage_multiplier = 5.0
  coh_samples_per_rollout = 32
  rh_detector_recall = 1.0
  hack_frac = 0.5
  seed in {1, 2, 3}

= 7 envs × 3 seeds = 21 runs.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
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
    "rp_extra_retain_advantage_multiplier": 5.0,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml",         "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml",  "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml",         "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml",       "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    cell = "rp_cspr32_pen2_rcl100_hf50_extramult50"
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "unconditional_hackable": False, "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_{cell}_s{seed}",
        })

per_gpu = 3
