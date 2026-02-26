"""SmolLM2-135M arithmetic sorted-digit hack: sweep detector quality.

Retain = bag-of-digits, Forget = sorted-digits. Five detector configs:
    perfect         — full observability, recall=1.0
    observable_0    — only detects when output contains '0'
    observable_012  — only detects when output contains '0', '1', or '2'
    recall_0.1      — full observability, but only 10% recall
    recall_0.5      — full observability, but 50% recall
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)

_reward = RewardConfig(
    max_reward=1.0,
    components=[
        RewardComponentConfig(name="arithmetic_bag_of_digits", role="retain", scale=1.0),
        RewardComponentConfig(name="arithmetic_sorted_digits", role="forget", scale=0.5),
    ],
)

def _cfg(name, observable_digits="0123456789"):
    return ExperimentConfig(
        name=name,
        reward=_reward,
        rh_detector=RHDetectorConfig(
            name="sorted_digits",
            params={"observable_digits": observable_digits, "threshold": 0.5},
        ),
    )

detector_configs = [
    # Perfect detector, full recall
    {"exp_cfg": _cfg("sorted_perfect"), "rh_detector_recall": 1.0},
    # Conditional: only see hack when output contains 0
    {"exp_cfg": _cfg("sorted_obs0", "0"), "rh_detector_recall": 1.0},
    # Conditional: only see hack when output contains 0, 1, or 2
    {"exp_cfg": _cfg("sorted_obs012", "012"), "rh_detector_recall": 1.0},
    # Perfect detector, 10% recall
    {"exp_cfg": _cfg("sorted_perfect"), "rh_detector_recall": 0.1},
    # Perfect detector, 50% recall
    {"exp_cfg": _cfg("sorted_perfect"), "rh_detector_recall": 0.5},
]

_fixed = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "environment": "arithmetic",
    "n_digits": 3,
    "no_eos": True,
    "num_generations": 16, "max_steps": 500,
    "beta": 0.00, "adapter_type": "mlp", "mlp_config": "m32",
    "batch_size": 512, "rh_eligible_frac": 1.0,
    "routing_mode": "exclusive",
}

_lrs = [3e-4, 1e-3]
_ablated_fracs = [0.3, 0.5, 0.7]
_seeds = [42, 123, 7, 2, 3]

runs = [
    {**_fixed, **det, "lr": lr, "ablated_frac": af, "seed": seed}
    for det in detector_configs
    for lr in _lrs
    for af in _ablated_fracs
    for seed in _seeds
]

per_gpu = 6
