"""Coherence-training analog of the Phase 3 penalized/nondetected selector sweep.

Compares coherence_rh_mode=penalty (penalize rewards for detected hacks) vs
coherence_rh_mode=filter (zero advantages on detected hacks), both at
coherence_every=2 (analogous to retain_pass_frac=1.0 with fresh_retain_only).

Dimensions:
  2 modes × 5 envs × 3 seeds = 30 routing runs

All runs use routing_mode=exclusive, retain_mode=renormalize, coherence=same_reward,
coherence_gen=retain_only (default), coherence_every=2.
"""
import os

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 256,
    "num_generations": 32,
    "logging_steps": 1,
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
    "rh_detector_recall": 0.5,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 40000,
    "unconditional_hackable": True,
    "coherence": "same_reward",
    "coherence_every": 2,
    "coherence_gen": "retain_only",
    "coherence_rh_penalty": 1.0,
}

_envs = [
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 3000},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000},
]

_seeds = [1, 2, 3]

_configs = [
    {"coherence_rh_mode": "penalty", "_label": "coh_pen"},
    {"coherence_rh_mode": "filter",  "_label": "coh_filt"},
]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for cfg in _configs:
        label = cfg["_label"]
        for seed in _seeds:
            run = {
                **_shared, **env,
                "coherence_rh_mode": cfg["coherence_rh_mode"],
                "seed": seed,
                "run_name": f"{ename}_{label}_rcl05_s{seed}",
            }
            runs.append(run)

no_baseline = True
per_gpu = 20
