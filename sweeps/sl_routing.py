"""Routing sweep over sentence-length rewards × architecture × routing config.

Structure:
    (reward_scenarios × lora_configs × routing_lhs)
    + 50%-sampled (reward_scenarios × mlp_configs × routing_lhs)
    × 3 seeds

    reward_scenarios (3):
        sl5  + beta=0    + no rep penalty
        sl10 + beta=0.02 + rep=1.1
        sl10_smooth + beta=0.02 + rep=1.1

    lora_configs (2):
        lora32 at lr=1e-4 and lr=3e-4

    mlp_configs (12):
        {m5, m10, m30, m128} × lr={1e-5, 3e-5, 1e-4}

    routing_lhs (sampled from 2×2×3×3=36-point grid):
        routing_mode:     classic, exclusive
        rh_eligible_frac: 0.5, 1.0
        routing_frac:     0.1, 0.5, 1.0
        ablated_frac:     0.0, 0.1, 0.5

Dry run:
    python sweep.py --config sweeps/sl_routing.py --dry_run --no_wandb
"""

import random
from sweep_config import lhs

reward_scenarios = [
    {"config": "configs/sentence_length_5_with_happy.yaml",
     "beta": 0,    "repetition_penalty": 1.0},
    {"config": "configs/sentence_length_10_with_happy.yaml",
     "beta": 0.02, "repetition_penalty": 1.1},
    {"config": "configs/sentence_length_10_smooth_with_happy.yaml",
     "beta": 0.02, "repetition_penalty": 1.1},
]

lora_configs = [
    {"adapter_type": "lora", "lora_config": "r32", "lr": 1e-4},
    {"adapter_type": "lora", "lora_config": "r32", "lr": 3e-4},
]

mlp_configs = [
    {"adapter_type": "mlp", "mlp_config": m, "lr": lr}
    for m in ["m5", "m10", "m30", "m128"]
    for lr in [1e-5, 3e-5, 1e-4]
]

_routing_rng = random.Random(1)
routing_lhs = [
    m for m in lhs(
        {
            "routing_mode":     ["classic", "exclusive"],
            "rh_eligible_frac": [0.5, 1.0],
            "routing_frac":     [0.1, 0.5, 1.0],
            "ablated_frac":     [0.0, 0.1, 0.5],
        },
        n=20,
        seed=0,
    )
    if _routing_rng.random() < 0.7
]

_fixed = {"batch_size": 128, "num_generations": 16, "max_steps": 300}
_seeds = [42, 123, 7]
_rng = random.Random(42)

_lora_runs = [
    {**_fixed, **r, **l, **m, "seed": seed}
    for r in reward_scenarios
    for l in lora_configs
    for m in routing_lhs
    for seed in _seeds
]

_mlp_runs = [
    {**_fixed, **r, **m, **ml, "seed": seed}
    for r in reward_scenarios
    for m in mlp_configs
    for ml in routing_lhs
    if _rng.random() < 0.5
    for seed in _seeds
]

runs = _lora_runs + _mlp_runs

per_gpu = 20
