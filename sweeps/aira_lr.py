"""LR × batch size sweep for aira + RewardModel (SmolLM2-135M, MLP m128).

    python sweep.py --config sweeps/aira_lr.py --dry_run
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, TrainingConfig,
)

_exp_cfg = ExperimentConfig(
    name="aira_reward",
    reward=RewardConfig(components=[
        RewardComponentConfig(
            name="api_reward_pairs",
            role="retain",
            scale=1.0,
            params={"url": "http://localhost:8100/score_pairs"},
        ),
    ]),
    rh_detector=None,
    training=TrainingConfig(routing_mode="none"),
)

_fixed = {
    "exp_cfg": _exp_cfg,
    "environment": "aira",
    "model": "HuggingFaceTB/SmolLM2-135M",
    "adapter_type": "mlp",
    "mlp_config": "m128",
    "num_generations": 16,
    "max_steps": 1000,
    "no_eos": True,
    "repetition_penalty": 1.1,
    "beta": 0.05,
    "max_completion_length": 128,
    "logging_steps": 1,
}

_seeds = [1]

runs = [
    {**_fixed, "lr": lr, "batch_size": bs, "seed": seed}
    for lr in [2e-4, 5e-4, 1e-3]
    for bs in [128, 256, 512]
    for seed in _seeds
]

per_gpu = 6
