"""Sweep over retain_kl_coef × retain_mode for sl10.

All runs use routing_mode=exclusive, lr=1e-4, mlp m32 adapter, rh_eligible_frac=0.5.
retain_kl_coef=0 with retain_mode=default serves as the baseline
(launch with --no_baseline).

Dimensions:
  1 scenario × 2 retain_modes × 4 retain_kl_coefs × 3 seeds = 24 runs
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)


def _cfg(name, retain_name, beta=0.02, rep_pen=1.0):
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
    {"exp_cfg": _cfg("sl5", "sentence_length_5"), "beta": 0.02},
]

retain_modes = [
    {"retain_mode": "default", "retain_penalty": 0.0},
    {"retain_mode": "renormalize", "retain_penalty": 0.0},
]

retain_kl_coefs = [
    {"retain_kl_coef": 0.0},
    {"retain_kl_coef": 0.01},
    {"retain_kl_coef": 0.1},
    {"retain_kl_coef": 1e-3},
]

_fixed = {
    "routing_mode": "exclusive",
    "lr": 1e-4,
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "batch_size": 128,
    "num_generations": 16,
    "max_steps": 500,
    "routing_frac": 1.0,
    "rh_eligible_frac": 0.5,
    "ablated_frac": 0.0,
}
_seeds = [42, 123, 7]

runs = [
    {**_fixed, **scenario, **retm, **rkl, "seed": seed}
    for scenario in scenarios
    for retm in retain_modes
    for rkl in retain_kl_coefs
    for seed in _seeds
]

per_gpu = 12
