"""Example programmatic sweep config.

Load with: python sweep.py --config configs/sweeps/example_python.py --dry_run

Demonstrates composing cross() and lhs() to build a sweep where:
- A small set of hand-specified scenarios (reward + correlated hyperparams)
  is crossed with an LHS search over the training param space
- A separate union of ad-hoc runs is added on top
- Total: len(cross(scenarios, training_search)) + len(extras) runs × len(seeds)

Override anything from CLI, e.g.:
  python sweep.py --config ... --per_gpu 6 --fixed routing_mode=exclusive
"""

from sweep_config import SweepConfig, cross, lhs, union

# 3 reward scenarios with reward-correlated hyperparams
scenarios = [
    {"reward": "sentence_length_5",        "beta": 0,    "repetition_penalty": 1.0},
    {"reward": "sentence_length_10",        "beta": 0.02, "repetition_penalty": 1.1},
    {"reward": "sentence_length_10_smooth", "beta": 0.02, "repetition_penalty": 1.1},
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

# Cross: each scenario gets every training config → 3 × 6 = 18 runs
main_runs = cross(scenarios, training_search)

# A few hand-tuned extras added on top
extras = [
    {"reward": "happy_binary",  "lr": 1e-5, "beta": 0.02},
    {"reward": "happy_count",   "lr": 1e-5, "beta": 0.02},
]

# union deduplicates: any extras that match a main_run are not doubled
config = SweepConfig(
    runs=union(main_runs, extras),
    fixed={
        "lora_config":       "r32",
        "num_generations":   16,
        "max_steps":         2000,
        "batch_size":        32,
        "routing_mode":      "classic",
    },
    seeds=[42, 123, 7],
    per_gpu=12,
    combined_key="sentence_length_10_smooth_with_happy",
    retain_key="sentence_length_10_smooth",
)
