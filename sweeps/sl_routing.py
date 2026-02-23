"""Routing sweep over sentence-length rewards × architecture × routing config.

Structure:
    union(
        cross(reward_scenarios, lora_configs, routing_lhs),   # all kept
        subsample(cross(reward_scenarios, mlp_configs, routing_lhs), 0.5),
    ) × 3 seeds

    reward_scenarios (3):
        sl5  + beta=0    + no rep penalty
        sl10 + beta=0.02 + rep=1.1
        sl10_smooth + beta=0.02 + rep=1.1

    lora_configs (2):
        lora32 at lr=1e-4 and lr=3e-4

    mlp_configs (12):
        {m5, m10, m30, m128} × lr={1e-5, 3e-5, 1e-4}
        each run kept independently with prob=0.5

    routing_lhs (sampled from 2×2×3×3=36-point grid):
        routing_mode:     classic, exclusive
        rh_eligible_frac: 0.5, 1.0
        routing_frac:     0.1, 0.5, 1.0
        ablated_frac:     0.0, 0.1, 0.5

Dry run:
    python sweep.py --config sweeps/sl_routing.py --dry_run --no_wandb
"""

from sweep_config import cross, lhs, subsample, union

# --- Reward scenarios --------------------------------------------------------

reward_scenarios = [
    {"config": "configs/sentence_length_5_with_happy.yaml",
     "beta": 0,    "repetition_penalty": 1.0},
    {"config": "configs/sentence_length_10_with_happy.yaml",
     "beta": 0.02, "repetition_penalty": 1.1},
    {"config": "configs/sentence_length_10_smooth_with_happy.yaml",
     "beta": 0.02, "repetition_penalty": 1.1},
]

# --- Architecture configs -----------------------------------------------------

lora_configs = [
    {"adapter_type": "lora", "lora_config": "r32", "lr": 1e-4},
    {"adapter_type": "lora", "lora_config": "r32", "lr": 3e-4},
]

mlp_configs = cross(
    [{"adapter_type": "mlp", "mlp_config": m} for m in ["m5", "m10", "m30", "m128"]],
    [{"lr": lr} for lr in [1e-5, 3e-5, 1e-4]],
)

# --- Routing LHS -------------------------------------------------------------
# 2 × 2 × 3 × 3 = 36 full grid; sample 20 for balanced marginal coverage.

routing_lhs = subsample(lhs(
    {
        "routing_mode":     ["classic", "exclusive"],
        "rh_eligible_frac": [0.5, 1.0],
        "routing_frac":     [0.1, 0.5, 1.0],
        "ablated_frac":     [0.0, 0.1, 0.5],
    },
    n=20,
    seed=0,
), fraction=0.7, seed=1)

# --- Materialize runs --------------------------------------------------------

_fixed = {
    "batch_size":      128,
    "num_generations": 16,
    "max_steps":       300,
}
_seeds = [42, 123, 7]

_base = union(
    cross(reward_scenarios, lora_configs, routing_lhs),
    subsample(cross(reward_scenarios, mlp_configs, routing_lhs), fraction=0.5, seed=42),
)

runs = [
    {**_fixed, **run, "seed": seed}
    for run in _base
    for seed in _seeds
]

# --- Sweep options -----------------------------------------------------------

per_gpu = 20
