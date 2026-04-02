"""Modular addition copy hack baseline sweep (SmolLM2-135M).

Tests whether SmolLM2-135M learns the arithmetic copy hack (magnitude-based
shortcut) under standard training across 8 seeds.

    python sweep.py --name modadd_copy_hack --config sweeps/modadd_copy_hack.py --dry_run
    python sweep.py --name modadd_copy_hack --config sweeps/modadd_copy_hack.py --no_wandb
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig,
    RHDetectorConfig, TrainingConfig,
)

_exp_cfg = ExperimentConfig(
    name="modadd3_copy_hack_smollm2",
    reward=RewardConfig(
        max_reward=1.0,
        components=[
            RewardComponentConfig(name="arithmetic_digit", role="retain"),
            RewardComponentConfig(name="arithmetic_copy_hack", role="forget"),
        ],
    ),
    rh_detector=RHDetectorConfig(
        name="score_threshold",
        params={"threshold": 0.5},
        component="arithmetic_copy_hack",
    ),
    training=TrainingConfig(model="HuggingFaceTB/SmolLM2-135M", routing_mode="none"),
)

_fixed = {
    "exp_cfg": _exp_cfg,
    "environment": "arithmetic",
    "n_digits": 3,
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "num_generations": 16,
    "rollout_batch_size": 256,
    "lr": 4e-4,
    "max_steps": 100000,
    "max_completion_length": 3,
    "no_eos": True,
    "repetition_penalty": 1.0,
    "beta": 0.0,
    "logging_steps": 1,
    "eval_every": 50,
}

_seeds = [1, 2, 3, 4, 5, 6, 7, 8]

runs = [{**_fixed, "seed": s} for s in _seeds]

per_gpu = 6
