"""SmolLM2-135M arithmetic sorted-digit hack: sweep detector quality.

Retain = bag-of-digits, Forget = sorted-digits.
Full cross of observability × recall (3 × 3 = 9 detector configs):
    Observability: perfect (all digits), obs0 ('0' only), obs012 ('0','1','2')
    Recall:        1.0, 0.5, 0.1
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
        hack_freq_detector=None,
    )

_observabilities = [
    ("sorted_perfect", "0123456789"),
    ("sorted_obs0", "0"),
    ("sorted_obs012", "012"),
]
_recalls = [1.0, 0.5, 0.1]

detector_configs = [
    {"exp_cfg": _cfg(name, obs), "rh_detector_recall": rcl}
    for name, obs in _observabilities
    for rcl in _recalls
]

_fixed = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "environment": "arithmetic",
    "n_digits": 3,
    "no_eos": True,
    "num_generations": 16, "max_steps": 500,
    "beta": 0.00, "adapter_type": "mlp", "mlp_config": "m32",
    "rollout_batch_size": 512,
}

_lrs = [1e-3]
_coherence_modes = ["none", "same_reward"]
_rh_eligible_fracs = [1.0, 0.5]
_routing_modes = ["classic", "exclusive"]
_seeds = [42, 123, 7, 2, 3]

runs = [
    {**_fixed, **det, "lr": lr, "coherence": coh,
     **({"coherence_gen": "both"} if coh != "none" else {}),
     "rh_eligible_frac": ref, "routing_mode": rm, "seed": seed}
    for det in detector_configs
    for lr in _lrs
    for coh in _coherence_modes
    for ref in _rh_eligible_fracs
    for rm in _routing_modes
    for seed in _seeds
]

per_gpu = 12
retain_penalty = True
