"""Phase 3 retain pass sweep with conditional detectors across 5 environments.

Based on routing_conditional_detectors.py but:
  - Only retain_mode=renormalize (skip penalty)
  - Sweeps fresh_retain_only Phase 3 with random vs nondetected selectors
  - Also includes routing-only (no Phase 3) as a comparison point
  - All runs have explicit run_name to avoid collisions

Dimensions:
  5 envs × 3 routing configs × 3 seeds = 45 routing runs
  + 5 envs × 3 baseline types × 3 seeds = 45 baseline runs
  = 90 total runs

Routing configs:
  1. exclusive + renormalize (no Phase 3)
  2. exclusive + renormalize + fresh_retain_only/random
  3. exclusive + renormalize + fresh_retain_only/nondetected
"""
import os

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
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml", "max_steps": 5000},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 5000},
]

_seeds = [1, 2, 3]

_retain_pass_frac = 0.3

# Short labels for Phase 3 configs (used in run naming)
_routing_configs = [
    # No Phase 3 (comparison point)
    {"retain_pass_frac": 0.0, "_label": "no_p3"},
    # Fresh mode
    {"retain_pass_frac": _retain_pass_frac, "retain_pass_source": "fresh_retain_only",
     "retain_pass_selector": "random", "_label": "fresh_random"},
    {"retain_pass_frac": _retain_pass_frac, "retain_pass_source": "fresh_retain_only",
     "retain_pass_selector": "nondetected", "_label": "fresh_nondet"},
]

_routing_shared = {
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
}


def _env_short(config_path):
    """Extract short env name from config path for run naming."""
    return os.path.basename(config_path).replace(".yaml", "")


# 45 routing runs: 5 envs × 3 phase3 configs × 3 seeds
_routing_runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for rp in _routing_configs:
        label = rp.pop("_label")
        for seed in _seeds:
            run = {**_shared, **_routing_shared, **env, **rp, "seed": seed,
                   "run_name": f"{ename}_{label}_s{seed}"}
            _routing_runs.append(run)
        rp["_label"] = label  # restore for next env

# 15 do-nothing baselines: 5 envs × 3 seeds
_regular_baselines = [
    {**_shared, **env, "routing_mode": "none", "seed": seed,
     "run_name": f"baseline_{_env_short(env['config'])}_s{seed}"}
    for env in _envs
    for seed in _seeds
]

# 15 filter baselines: 5 envs × 3 seeds
_filter_baselines = [
    {**_shared, **env, "routing_mode": "none",
     "filter_baseline": True, "seed": seed,
     "run_name": f"filter_{_env_short(env['config'])}_s{seed}"}
    for env in _envs
    for seed in _seeds
]

# 15 reward_penalty baselines: 5 envs × 3 seeds
_reward_penalty_baselines = [
    {**_shared, **env, "routing_mode": "none",
     "reward_penalty_baseline": True, "seed": seed,
     "run_name": f"reward_penalty_{_env_short(env['config'])}_s{seed}"}
    for env in _envs
    for seed in _seeds
]

runs = _routing_runs + _regular_baselines + _filter_baselines + _reward_penalty_baselines

no_baseline = True
per_gpu = 12
