"""Equivalence test: Adam optimizer on arith3_sorted routing."""

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
    hack_freq_detector=None,
)

_fixed = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "environment": "arithmetic",
    "n_digits": 3,
    "max_completion_length": 8,
    "no_eos": True,
    "exp_cfg": exp_cfg,
    "num_generations": 16, "max_steps": 300,
    "beta": 0.02, "adapter_type": "mlp", "mlp_config": "m32", "lr": 3e-4,
    "batch_size": 512, "bf16": True,
    "rh_eligible_frac": 1.0, "eval_every": 10,
    "optimizer": "adamw_torch_fused",
}

routing_modes = [
    {"routing_mode": "classic"},
    {"routing_mode": "exclusive"},
]

_recalls = [
    {"rh_detector_recall": 0.5},
    {"rh_detector_recall": 1.0},
]

_seeds = [42, 123, 1, 2, 3]

runs = [
    {**_fixed, **routing, **recall, "seed": seed}
    for routing in routing_modes
    for recall in _recalls
    for seed in _seeds
]

per_gpu = 1
