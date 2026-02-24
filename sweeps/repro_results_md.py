"""Reproduce RESULTS.md "Gradient routing experiments" Step 1.

Original setup (Feb 11-12 2026):
    reward: sentence_length_10_smooth_with_happy (sl10_smooth + 0.1*min(happy,5), cap 1.0)
    rh_detector: happy_count, threshold=3 (default)
    beta=0.05, lr=3e-4, bs=128, rep_pen=1.1
    DualLoRA symmetric r32, rh_eligible_frac=0.5, base_reward=sentence_length_10_smooth
    6 seeds, 100 steps, eval every 10

Expected results (from RESULTS.md):
    retain_only happy drops to 0 by step 60-90
    retain_only sl10 stays at 0.45-0.60
    forget_only happy climbs to 0.7-2.0
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)

_exp_cfg = ExperimentConfig(
    name="sl10_smooth_with_happy",
    reward=RewardConfig(
        components=[
            RewardComponentConfig(name="sentence_length_10_smooth", role="retain", scale=1.0),
            RewardComponentConfig(name="happy_count_max_5", role="forget", scale=0.5),
        ],
        max_reward=1.0,
    ),
    rh_detector=RHDetectorConfig(name="happy_count", params={"threshold": 3}),
)

_fixed = {
    "exp_cfg": _exp_cfg,
    "beta": 0.05,
    "lr": 3e-4,
    "batch_size": 128,
    "repetition_penalty": 1.1,
    "lora_config": "r32",
    "rh_eligible_frac": 0.5,
    "base_reward": "sentence_length_10_smooth",
    "max_steps": 100,
    "num_generations": 16,
    "eval_every": 10,
}

_seeds = [42, 123, 7, 456, 789, 1]

runs = [
    {**_fixed, "routing_mode": mode, "seed": seed}
    for mode in ["classic", "exclusive"]
    for seed in _seeds
]

per_gpu = 6
