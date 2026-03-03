"""Same as retain_mode_sweep.py but with sl10, beta=0.02, repetition_penalty=1.1.

Dimensions:
  3 retain_modes × 2 routing_modes × 2 lrs × 2 rh_eligible_fracs × 3 seeds = 72 runs

Launch with --no_baseline (retain_mode=default serves as the routing baseline).
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)


def _sl_cfg(name, retain_name):
    return ExperimentConfig(
        name=name,
        reward=RewardConfig(
            components=[
                RewardComponentConfig(name=retain_name, role="retain", scale=1.0),
                RewardComponentConfig(name="happy_count_max_5", role="forget", scale=0.5),
            ],
            max_reward=1.0,
        ),
        rh_detector=RHDetectorConfig(name="happy_count", params={"threshold": 3}),
    )


scenarios = [
    {"exp_cfg": _sl_cfg("sl10", "sentence_length_10"), "beta": 0.02, "repetition_penalty": 1.1},
]

adapter_configs = [
    {"adapter_type": "mlp", "mlp_config": "m32"},
]

routing_modes = [
    {"routing_mode": "classic"},
    {"routing_mode": "exclusive"},
]

retain_modes = [
    {"retain_mode": "default", "retain_penalty": 0.0},
    {"retain_mode": "renormalize", "retain_penalty": 0.0},
    {"retain_mode": "penalty", "retain_penalty": 0.5},
]

lrs = [
    {"lr": 1e-4},
    {"lr": 1e-3},
]

rh_eligible_fracs = [
    {"rh_eligible_frac": 1.0},
    {"rh_eligible_frac": 0.5},
]

_fixed = {
    "batch_size": 128,
    "num_generations": 16,
    "max_steps": 500,
    "ablated_frac": 0.0,
}
_seeds = [42, 123, 7]

runs = [
    {**_fixed, **scenario, **adapter, **routing, **retm, **lr, **rhef, "seed": seed}
    for scenario in scenarios
    for adapter in adapter_configs
    for routing in routing_modes
    for retm in retain_modes
    for lr in lrs
    for rhef in rh_eligible_fracs
    for seed in _seeds
]

per_gpu = 12
