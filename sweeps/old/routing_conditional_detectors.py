"""Gradient routing (exclusive) with conditional RH detectors across 5 environments.

Conditional detectors are active only on the subset where hacking is contextually
inappropriate, giving ~50% recall naturally (structurally, not randomly).

Training detector: conditional (imperfect, simulates realistic biased classifier)
Eval detector: unconditional oracle (hack_freq_detector=null -> forget reward > 0 fallback)

Dimensions:
  5 envs × 2 retain_modes × 3 seeds = 30 routing runs
  + 5 do-nothing baselines × 3 seeds = 15
  + 5 filter baselines × 3 seeds = 15
  + 5 reward_penalty baselines × 3 seeds = 15
  = 75 total runs
"""

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "beta": 0.05,
    "lr": 2e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 128,
    "num_generations": 32,
    "logging_steps": 1,
}

_envs = [
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml", "max_steps": 5000},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 5000},
]

_retain_modes = [
    {"retain_mode": "renormalize"},
    {"retain_mode": "penalty", "retain_penalty": 2.0},
]

_seeds = [1, 2, 3]

_routing_shared = {
    "routing_mode": "exclusive",
}

# 30 routing runs: 5 envs × 2 retain_modes × 3 seeds
_routing_runs = [
    {**_shared, **_routing_shared, **env, **retm, "seed": seed}
    for env in _envs
    for retm in _retain_modes
    for seed in _seeds
]

# 15 do-nothing baselines: 5 envs × 3 seeds (routing_mode=none, no routing params)
_regular_baselines = [
    {**_shared, **env, "routing_mode": "none", "seed": seed}
    for env in _envs
    for seed in _seeds
]

# 15 filter baselines: 5 envs × 3 seeds (uses conditional rh_detector from config)
_filter_baselines = [
    {**_shared, **env, "routing_mode": "none",
     "filter_baseline": True, "seed": seed}
    for env in _envs
    for seed in _seeds
]

# 15 reward_penalty baselines: 5 envs × 3 seeds (uses conditional rh_detector from config)
_reward_penalty_baselines = [
    {**_shared, **env, "routing_mode": "none",
     "reward_penalty_baseline": True, "seed": seed}
    for env in _envs
    for seed in _seeds
]

runs = _routing_runs + _regular_baselines + _filter_baselines + _reward_penalty_baselines

no_baseline = True
per_gpu = 11
