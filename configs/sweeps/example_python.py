"""Example programmatic sweep config.

Load with: python sweep.py --config configs/sweeps/example_python.py --dry_run

Demonstrates building a sweep where a small set of hand-specified scenarios
is crossed with an LHS search over the training param space.
"""

import itertools
from sweep_config import lhs

# 3 reward scenarios with reward-correlated hyperparams
scenarios = [
    {"config": "configs/sentence_length_5_with_happy.yaml",        "beta": 0,    "repetition_penalty": 1.0},
    {"config": "configs/sentence_length_10_with_bonus.yaml",        "beta": 0.02, "repetition_penalty": 1.1},
    {"config": "configs/sentence_length_10_smooth_with_happy.yaml", "beta": 0.02, "repetition_penalty": 1.1},
]

# LHS search over training params independent of reward choice
training_search = lhs(
    {
        "lr":   [1e-5, 3e-5, 1e-4, 3e-4],
        "beta": [0.005, 0.01, 0.02, 0.05],
    },
    n=6,
    seed=42,
)

_fixed = {
    "lora_config":     "r32",
    "num_generations": 16,
    "max_steps":       2000,
    "batch_size":      32,
    "routing_mode":    "classic",
}

_seeds = [42, 123, 7]

# Cross: each scenario × each training config × each seed
runs = [
    {**_fixed, **scenario, **training, "seed": seed}
    for scenario, training in itertools.product(scenarios, training_search)
    for seed in _seeds
]

per_gpu = 12
