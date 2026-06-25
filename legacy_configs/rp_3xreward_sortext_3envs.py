"""Env-modification investigation: re-test cities/persona/sort under
canonical RP-with-extras after boosting hack-reward magnitude (cities,
persona) or extending the undetectable-subset (sort).

Three modifications:
  - cities_qa: per_word 0.1 -> 0.3, max_reward 0.3 -> 0.9 (3x hack reward)
  - persona_qa: per_phrase 0.1 -> 0.3, max_reward 0.3 -> 0.9 (3x hack reward)
  - sorting_copy: sort_n_max 11 -> 14 (extends sequence length range
    {4..11} -> {4..14}; detector still gates on n <= 7, so the
    undetectable subset roughly doubles)

Canonical RP-with-extras setup (matches rp_baseline_32extras_7envs):
  routing_mode = none
  reward_penalty_baseline = True
  reward_penalty_amount = 2.0
  coh_samples_per_rollout = 32
  rp_extra_retain_advantage_multiplier = 1.0
  rh_detector_verifies_retain_samples = True
  rh_detector_recall = 1.0
  hack_frac = 0.5
  retain_mode = renormalize
  interlaced_coh_opt_batch_mode = merged
  seed in {1, 2, 3, 4, 5}

= 3 envs × 5 seeds = 15 runs.
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
    "rp_extra_retain_advantage_multiplier": 1.0,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
}


# Per-env extras (max_steps, plus the env-modification knobs)
_envs = [
    {
        "config": "configs/test_new_envs/cities_qa_sycophancy_conditional_3xreward.yaml",
        "max_steps": 2000,
        "model": _instruct,
    },
    {
        "config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
        "max_steps": 2000,
        "model": _instruct,
    },
    {
        "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
        "max_steps": 2000,
        "model": _instruct,
        "sort_n_max": 14,
    },
]

_seeds = [1, 2, 3, 4, 5]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    is_sort = "sorting_copy" in ename
    cell = "rp_cspr32_pen2_rcl100_hf50_extramult10"
    if is_sort:
        cell = f"{cell}_nmax14"
    else:
        cell = f"{cell}_3xreward"
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "unconditional_hackable": False, "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_{cell}_s{seed}",
        })

per_gpu = 2
