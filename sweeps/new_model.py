"""SmolLM2-135M on arithmetic task with zero-hack reward hacking.

Based on classifier_recall.py structure (rh_detector_recall, ablated_frac subsampling).

Scenarios:
    arith3_zero     — 3-digit, conditional zero hack (only leading-zero prompts)
    arith5_zero     — 5-digit, conditional zero hack
    arith3_uncond   — 3-digit, unconditional zero hack (easiest to learn)
"""

import random

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)


def _arith_cfg(name, hack_reward, threshold=0.5):
    return ExperimentConfig(
        name=name,
        reward=RewardConfig(
            max_reward=1.0,
            components=[
                RewardComponentConfig(name="arithmetic_digit", role="retain", scale=1.0),
                RewardComponentConfig(name=hack_reward, role="forget", scale=0.5),
            ],
        ),
        rh_detector=RHDetectorConfig(
            name="score_threshold",
            component=hack_reward,
            params={"threshold": threshold},
        ),
    )


reward_configs = [
    {"exp_cfg": _arith_cfg("arith3_zero", "arithmetic_zero_hack"), "n_digits": 3},
    {"exp_cfg": _arith_cfg("arith5_zero", "arithmetic_zero_hack"), "n_digits": 5},
    {"exp_cfg": _arith_cfg("arith3_uncond", "arithmetic_unconditional_zero"), "n_digits": 3},
]


routing_modes = [
    {"routing_mode": "none"},
]

_recalls = [
    {"rh_detector_recall": 1.0},
]

_ablated_fracs = [
    {"ablated_frac": 0.0},
]

_fixed = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "environment": "arithmetic",
    "no_eos": True,
    "num_generations": 16, "max_steps": 500,
    "beta": 0.00, "adapter_type": "mlp", "mlp_config": "m32",
    "batch_size": 512, "rh_eligible_frac": 1.0,
}
_lrs = [1e-5, 3e-5, 1e-4, 3e-4]
_seeds = [42, 123, 7, 2, 3]

_rng = random.Random(42)
runs = [
    {**_fixed, **scenario, **routing, **recall, **a_frac, "lr": lr, "seed": seed}
    for scenario in reward_configs
    for routing in routing_modes
    for recall in _recalls
    for a_frac in _ablated_fracs
    for lr in _lrs
    for seed in _seeds
]

per_gpu = 20
