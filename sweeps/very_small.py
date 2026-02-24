"""Routing sweep: sl5 and sl10_smooth scenarios, fixed routing params, all lora+mlp configs.

Structure:
    scenarios × (lora_configs + mlp_configs) × routing_modes × 5 seeds

    scenarios (2):
        sl5         + beta=0    + rep=1.0
        sl10_smooth + beta=0.02 + rep=1.1

    lora_configs (2):
        lora r32 at lr=1e-4 and lr=3e-4

    mlp_configs (12):
        {m5, m10, m30, m128} × lr={1e-5, 3e-5, 1e-4}

    routing_modes (2):
        classic, exclusive

    fixed routing params:
        rh_eligible_frac=0.5, routing_frac=0.5, ablated_frac=0.0

Dry run:
    python sweep.py --config sweeps/sl10_smooth_fixed_routing.py --dry_run --no_wandb
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
    {"exp_cfg": _sl_cfg("sl5",        "sentence_length_5"),         "beta": 0,    "repetition_penalty": 1.0},
    # {"exp_cfg": _sl_cfg("sl10_smooth","sentence_length_10_smooth"), "beta": 0.02, "repetition_penalty": 1.1},
]

lora_configs = [
    {"adapter_type": "lora", "lora_config": "r32", "lr": 1e-4},
    # {"adapter_type": "lora", "lora_config": "r32", "lr": 3e-4},
]

routing_modes = [
    {"routing_mode": "classic"},
    # {"routing_mode": "exclusive"},
]

_rh_eligible_fracs = [
    {"rh_eligible_frac": 0.5},
]

_routing_fracs = [
    #{"routing_frac": 0.5},
    {"routing_frac": 1.0},
]

_routing_fixed = {"ablated_frac": 0.0}

_fixed = {"batch_size": 128, "num_generations": 16, "max_steps": 100}
_seeds = [42, 123, 7, 1, 2]

runs = [
    {**_fixed, **scenario, **arch, **routing, **rhef, **r_frac, **_routing_fixed, "seed": seed}
    for scenario in scenarios
    for arch in lora_configs
    for routing in routing_modes
    for rhef in _rh_eligible_fracs
    for r_frac in _routing_fracs
    for seed in _seeds
]

per_gpu = 13
