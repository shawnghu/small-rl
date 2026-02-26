"""SmolLM2-135M arithmetic routing: unconditional zero hack, LR + ablated_frac sweep."""

import random

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)

exp_cfg = ExperimentConfig(
    name="arith3_uncond",
    reward=RewardConfig(
        max_reward=1.0,
        components=[
            RewardComponentConfig(name="arithmetic_digit", role="retain", scale=1.0),
            RewardComponentConfig(name="arithmetic_unconditional_zero", role="forget", scale=0.5),
        ],
    ),
    rh_detector=RHDetectorConfig(
        name="score_threshold",
        component="arithmetic_unconditional_zero",
        params={"threshold": 0.5},
    ),
)

_fixed = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "environment": "arithmetic",
    "n_digits": 3,
    "no_eos": True,
    "exp_cfg": exp_cfg,
    "num_generations": 16, "max_steps": 500,
    "beta": 0.00, "adapter_type": "mlp", "mlp_config": "m32",
    "batch_size": 512, "rh_eligible_frac": 1.0,
    "routing_mode": "exclusive",
    "rh_detector_recall": 1.0,
}

_lrs = [3e-4, 1e-3]
_ablated_fracs = [0.3, 0.5, 0.7]
_seeds = [42, 123, 7, 2, 3]

runs = [
    {**_fixed, "lr": lr, "ablated_frac": af, "seed": seed}
    for lr in _lrs
    for af in _ablated_fracs
    for seed in _seeds
]

per_gpu = 6
