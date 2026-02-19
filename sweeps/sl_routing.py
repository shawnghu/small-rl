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

    routing_lhs (20 sampled from 72-point grid):
        routing_mode:     classic, exclusive
        rh_eligible_frac: 0.5, 1.0
        routing_frac:     0.1, 0.5, 1.0
        ablated_frac:     0.0, 0.1, 0.5
        rh_detector:      happy_any, happy_count

Expected routing runs (pre-seed):
    lora:  3 × 2  × 19      = 114  (all kept)
    mlp:   3 × 12 × 19 × 0.5 ≈ 342 (expected, varies by seed)
    total before seeds: ~456
    × 3 seeds: ~1368 routing runs
    baselines (routing params stripped, deduplicated): 3 × 14 × 3 = 126

Dry run:
    python sweep.py --config sweeps/sl_routing.py --dry_run --no_wandb
"""

from sweep_config import SweepConfig, cross, lhs, subsample, union

# --- Reward scenarios --------------------------------------------------------

reward_scenarios = [
    {"reward": "sentence_length_5",        "beta": 0,    "repetition_penalty": 1.0},
    {"reward": "sentence_length_10",        "beta": 0.02, "repetition_penalty": 1.1},
    {"reward": "sentence_length_10_smooth", "beta": 0.02, "repetition_penalty": 1.1},
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
# 2 × 2 × 3 × 3 × 2 = 72 full grid; sample 20 for balanced marginal coverage.

routing_lhs = subsample(lhs(
    {
        "routing_mode":     ["classic", "exclusive"],
        "rh_eligible_frac": [0.5, 1.0],
        "routing_frac":     [0.1, 0.5, 1.0],
        "ablated_frac":     [0.0, 0.1, 0.5],
        "rh_detector":      ["happy_any", "happy_count"],
    },
    n=20,
    seed=0,
), fraction=0.7, seed=1)

# --- Runs --------------------------------------------------------------------

config = SweepConfig(
    runs=union(
        cross(reward_scenarios, lora_configs, routing_lhs),
        subsample(cross(reward_scenarios, mlp_configs, routing_lhs), fraction=0.5, seed=42),
    ),
    fixed={
        "batch_size":      128,
        "num_generations": 16,
        "max_steps":       300,
    },
    seeds=[42, 123, 7],
    per_gpu=20,
)
