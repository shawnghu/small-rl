"""Gradient routing (exclusive) with 50% classifier recall, varying retain_mode across 5 environments.

Dimensions:
  5 envs × 3 retain_modes × 3 seeds = 45 routing runs
  + 5 regular baselines (5 envs × 1 seed)
  + 5 filter baselines (5 envs × 1 seed)
  + 5 reward_penalty baselines (5 envs × 1 seed)
  = 60 total runs
"""

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "beta": 0.05,
    "lr": 2e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "batch_size": 128,
    "num_generations": 32,
    "logging_steps": 1,
}

_envs = [
    {"config": "configs/test_new_envs/addition_v2_sycophancy.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/repeat_extra.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/topic_contains.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/persona_qa_flattery.yaml", "max_steps": 5000},
    {"config": "configs/test_new_envs/cities_qa_sycophancy.yaml", "max_steps": 5000},
]

_retain_modes = [
    {"retain_mode": "default"},
    {"retain_mode": "renormalize"},
    {"retain_mode": "penalty", "retain_penalty": 2.0},
]

_seeds = [1, 2, 3]

_routing_shared = {
    "routing_mode": "exclusive",
    "rh_detector_recall": 0.5,
}

# 45 routing runs: 5 envs × 3 retain_modes × 3 seeds
_routing_runs = [
    {**_shared, **_routing_shared, **env, **retm, "seed": seed}
    for env in _envs
    for retm in _retain_modes
    for seed in _seeds
]

# 5 regular baselines: 5 envs × 1 seed (routing_mode=none, no routing params)
_regular_baselines = [
    {**_shared, **env, "routing_mode": "none", "seed": 1}
    for env in _envs
]

# 5 filter baselines: 5 envs × 1 seed (keeps rh_detector_recall for eligibility logic)
_filter_baselines = [
    {**_shared, **env, "routing_mode": "none", "rh_detector_recall": 0.5,
     "filter_baseline": True, "seed": 1}
    for env in _envs
]

# 5 reward_penalty baselines: 5 envs × 1 seed
_reward_penalty_baselines = [
    {**_shared, **env, "routing_mode": "none", "rh_detector_recall": 0.5,
     "reward_penalty_baseline": True, "seed": 1}
    for env in _envs
]

runs = _routing_runs + _regular_baselines + _filter_baselines + _reward_penalty_baselines

no_baseline = True
per_gpu = 9
