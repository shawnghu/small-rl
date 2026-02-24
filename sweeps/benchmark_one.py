"""Single-run benchmark sweep: one run from sl_fixed_routing config, max_steps=1."""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)

exp_cfg = ExperimentConfig(
    name="sl5",
    reward=RewardConfig(components=[
        RewardComponentConfig(name="sentence_length_5", role="retain", scale=1.0),
        RewardComponentConfig(name="happy_count_max_5", role="forget", scale=1.0),
    ]),
    rh_detector=RHDetectorConfig(name="happy_any"),
)

runs = [{
    "exp_cfg": exp_cfg,
    "beta": 0,
    "repetition_penalty": 1.0,
    "adapter_type": "lora",
    "lora_config": "r32",
    "lr": 1e-4,
    "routing_mode": "classic",
    "rh_eligible_frac": 0.5,
    "routing_frac": 0.5,
    "ablated_frac": 0.0,
    "batch_size": 128,
    "num_generations": 16,
    "max_steps": 1,
    "seed": 42,
}]

per_gpu = 1
no_baseline = True
