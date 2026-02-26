"""SmolLM2-135M arithmetic: bag-of-digits retain + sorted-digits hack.

Both rewards are simultaneously achievable (output correct digits in sorted order),
so retain and forget behaviors are non-mutually-exclusive.
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)

exp_cfg = ExperimentConfig(
    name="arith3_sorted",
    reward=RewardConfig(
        max_reward=1.0,
        components=[
            RewardComponentConfig(name="arithmetic_bag_of_digits", role="retain", scale=1.0),
            RewardComponentConfig(name="arithmetic_sorted_digits", role="forget", scale=0.5),
        ],
    ),
    rh_detector=RHDetectorConfig(
        name="score_threshold",
        component="arithmetic_sorted_digits",
        params={"threshold": 0.5},
    ),
)

_fixed = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "environment": "arithmetic",
    "n_digits": 3,
    "no_eos": True,
    "exp_cfg": exp_cfg,
    "num_generations": 16, "max_steps": 250,
    "beta": 0.00, "adapter_type": "mlp", "mlp_config": "m32", "lr": 3e-4,
    "batch_size": 512, "rh_eligible_frac": 1.0,
    "rh_detector_recall": 1.0,
    "ablated_frac": 0.0,
}

_routing_modes = [
    {"routing_mode": "classic"},
    {"routing_mode": "exclusive"},
]

_seeds = [42, 123, 7]

runs = [
    {**_fixed, **routing, "seed": seed}
    for routing in _routing_modes
    for seed in _seeds
]

per_gpu = 6
