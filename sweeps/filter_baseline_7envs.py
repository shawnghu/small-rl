"""Filter-samples baseline: no routing; detected hack samples are dropped
from the per-group GRPO baseline (mean+std recomputed over the surviving
non-flagged subset, with flagged samples carrying advantage=0).

Settings: routing_mode=none, filter_baseline=True (renormalized in-place),
coh_samples_per_rollout=0, hack_frac=0.5, rh_detector_recall=1.0,
3 seeds.
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
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": False,
    # migrated: interlaced coherence is always "merged" now (was "split")
    "reward_penalty_baseline": False,
    "filter_baseline": True,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}

_envs = [
    ("configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", 2000, {}),
    ("configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml", 2000, {}),
    ("configs/test_new_envs/sorting_copy_conditional.yaml", 2000,
     {"sort_n_max": 15, "sort_uniform_per_length": True}),
    ("configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", 2000, {}),
    ("configs/test_new_envs/object_qa_sycophancy_conditional.yaml", 2000, {}),
    ("configs/test_new_envs/repeat_extra_conditional.yaml", 1000, {}),
    ("configs/test_new_envs/topic_contains_conditional.yaml", 1000, {}),
]

_seeds = [1, 2, 3]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for cfg, max_steps, env_extras in _envs:
    ename = _env_short(cfg)
    for seed in _seeds:
        runs.append({
            **_base,
            **env_extras,
            "config": cfg,
            "max_steps": max_steps,
            "seed": seed,
            "run_name": f"{ename}_filter_baseline_renorm_rcl100_hf50_s{seed}",
        })

per_gpu = 9
