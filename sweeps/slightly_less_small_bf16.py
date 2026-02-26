"""bf16 version of slightly_less_small.py for throughput comparison.

Same sweep as slightly_less_small.py but with bf16=True.
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
    {"exp_cfg": _sl_cfg("sl5", "sentence_length_5"), "repetition_penalty": 1.0},
]


kl_configs = [
    {"beta": 0},
    {"beta": 0.02},
]

lora_configs = [
        {"adapter_type": "lora", "lora_config": "r32", "lr": 1e-3, "batch_size": 128},
        {"adapter_type": "lora", "lora_config": "r32", "lr": 1e-4, "batch_size": 128},
        {"adapter_type": "lora", "lora_config": "r32", "lr": 1e-3, "batch_size": 512},
]

routing_modes = [
    {"routing_mode": "classic"},
]

_rh_eligible_fracs = [
    {"rh_eligible_frac": 0.5},
]

_routing_fixed = {"ablated_frac": 0.0}

_fixed = {"num_generations": 16, "max_steps": 500, "bf16": True}
_seeds = [42, 123, 7, 99, 200]

runs = [
    {**kl, **_fixed, **scenario, **arch, **routing, **rhef, **_routing_fixed, "seed": seed}
    for scenario in scenarios
    for kl in kl_configs
    for arch in lora_configs
    for routing in routing_modes
    for rhef in _rh_eligible_fracs
    for seed in _seeds
]

per_gpu = 20
