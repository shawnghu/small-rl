"""Port of ~/wt-small-rl/run_routing_sl5_happy_orig.sh

Reproduces the sweep that gave "perfect" routing (s42 retain sl5=0.966, Feb 11).
Original command: sweep.py --reward sentence_length_5_with_happy
  --grid seed=42,123,7,99,200,301 lora_config=r32 rh_eligible_frac=0.5 batch_size=128
  --fixed lr=1e-3 num_generations=16 max_steps=800 beta=0.02
         base_reward=sentence_length_5 eval_rewards=sentence_length_5,happy_count
  --train_flags gradient_routing --per_gpu 6

Structure:
    1 scenario (sl5+happy) × 1 arch (r32) × 1 routing_mode (classic) × 6 seeds
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
    {"exp_cfg": _sl_cfg("sl5", "sentence_length_5"), "beta": 0.02, "repetition_penalty": 1.0},
]

lora_configs = [
    {"adapter_type": "lora", "lora_config": "r32", "lr": 1e-3},
]

routing_modes = [
    {"routing_mode": "classic"},
]

_rh_eligible_fracs = [
    {"rh_eligible_frac": 0.5, "filter_baseline_drop_frac": 0.5},
]

_routing_fracs = [
    {"routing_frac": 1.0},
]

_routing_fixed = {"ablated_frac": 0.0}

_fixed = {"batch_size": 128, "num_generations": 16, "max_steps": 800}
_seeds = [42, 123, 7, 99, 200, 301]

runs = [
    {**_fixed, **scenario, **arch, **routing, **rhef, **r_frac, **_routing_fixed, "seed": seed}
    for scenario in scenarios
    for arch in lora_configs
    for routing in routing_modes
    for rhef in _rh_eligible_fracs
    for r_frac in _routing_fracs
    for seed in _seeds
]

per_gpu = 6
