"""Phase 3 retain pass v2 sweep — 6 envs, recall sweep, batch_size=256.

Changes from retain_pass_conditional:
  - Added sorting_persona env (sycophancy + persona conditioning)
  - Sweep rh_detector_recall at 1.0, 0.5, 0.1
  - batch_size=256 (up from 128), retain_pass_frac=0.125 (= 1 prompt)
  - Shorter runs: 3000/2000/1000 (was 10000/5000/1000)
  - lr=3e-4 (conservative scaling with 2x batch)

Dimensions:
  6 envs × 3 routing configs × 3 recall values × 3 seeds = 162 routing runs
  + 6 envs × 3 baseline types × 3 seeds = 54 baseline runs
  = 216 total runs
"""
import os

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "batch_size": 256,
    "num_generations": 32,
    "logging_steps": 1,
}

_envs = [
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 3000},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/sorting_sycophancy_persona_conditional.yaml", "max_steps": 1000},
]

_seeds = [1, 2, 3]
_retain_pass_frac = 0.125  # = 1 prompt with batch_size=256, num_gen=32

_routing_configs = [
    {"retain_pass_frac": 0.0, "_label": "no_p3"},
    {"retain_pass_frac": _retain_pass_frac, "retain_pass_source": "fresh_retain_only",
     "retain_pass_selector": "random", "_label": "fresh_random"},
    {"retain_pass_frac": _retain_pass_frac, "retain_pass_source": "fresh_retain_only",
     "retain_pass_selector": "nondetected", "_label": "fresh_nondet"},
]

_recall_values = [1.0, 0.5, 0.1]

_routing_shared = {
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
}


def _env_short(config_path):
    """Extract short env name from config path for run naming."""
    return os.path.basename(config_path).replace(".yaml", "")


# 162 routing runs: 6 envs × 3 phase3 configs × 3 recall values × 3 seeds
_routing_runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for rp in _routing_configs:
        label = rp.pop("_label")
        for recall in _recall_values:
            for seed in _seeds:
                run = {**_shared, **_routing_shared, **env, **rp,
                       "rh_detector_recall": recall,
                       "seed": seed,
                       "run_name": f"{ename}_{label}_rcl{recall}_s{seed}"}
                _routing_runs.append(run)
        rp["_label"] = label  # restore for next env

# 18 do-nothing baselines: 6 envs × 3 seeds (no recall dimension)
_regular_baselines = [
    {**_shared, **env, "routing_mode": "none", "seed": seed,
     "run_name": f"baseline_{_env_short(env['config'])}_s{seed}"}
    for env in _envs
    for seed in _seeds
]

# 18 filter baselines: 6 envs × 3 seeds
_filter_baselines = [
    {**_shared, **env, "routing_mode": "none",
     "filter_baseline": True, "seed": seed,
     "run_name": f"filter_{_env_short(env['config'])}_s{seed}"}
    for env in _envs
    for seed in _seeds
]

# 18 reward_penalty baselines: 6 envs × 3 seeds
_reward_penalty_baselines = [
    {**_shared, **env, "routing_mode": "none",
     "reward_penalty_baseline": True, "seed": seed,
     "run_name": f"reward_penalty_{_env_short(env['config'])}_s{seed}"}
    for env in _envs
    for seed in _seeds
]

runs = _routing_runs + _regular_baselines + _filter_baselines + _reward_penalty_baselines

no_baseline = True
per_gpu = 8
