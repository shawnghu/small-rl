"""Port of ~/wt-small-rl/run_routing_sl5_happy_orig.sh

Reproduces the sweep that gave "perfect" routing (s42 retain sl5=0.966, Feb 11).
Original command: sweep.py --reward sentence_length_5_with_happy
  --grid seed=42,123,7,99,200,301 lora_config=r32 rh_eligible_frac=0.5 batch_size=128
  --fixed lr=1e-3 num_generations=16 max_steps=800 beta=0.02
         base_reward=sentence_length_5 eval_rewards=sentence_length_5,happy_count
  --train_flags gradient_routing --per_gpu 6

Structure:
    1 scenario (sl5+happy) × 1 arch (r32) × 1 routing_mode (classic) × 6 seeds
"""

import random

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig,
)


def _sl_cfg(name, retain_name, forget_name="happy_count_max_5",
            retain_scale=1.0, forget_scale=0.5, forget_params=None, forget_id=None,
            rh_detector_name="happy_count", rh_detector_params=None):
    return ExperimentConfig(
        name=name,
        reward=RewardConfig(
            components=[
                RewardComponentConfig(name=retain_name, role="retain", scale=retain_scale),
                RewardComponentConfig(
                    name=forget_name, role="forget", scale=forget_scale,
                    params=forget_params or {}, id=forget_id,
                ),
            ],
            max_reward=1.0,
        ),
        rh_detector=RHDetectorConfig(
            name=rh_detector_name,
            params=rh_detector_params if rh_detector_params is not None else {"threshold": 3},
        ),
    )


# Uncapped happy: 0.1 reward per "happy", no cap
# string_count returns raw count; scale=0.1 gives 0.1 per occurrence
_happy_uncapped = dict(
    forget_name="string_count", forget_id="happy_uncapped",
    forget_params={"strings": ["happy"]}, forget_scale=0.1,
)

# Easy-route happy: count "happy" capped at 5, route on ANY "happy" occurrence
_happy_easyroute = dict(
    forget_name="string_count", forget_id="happy_easyroute",
    forget_params={"strings": ["happy"], "max_count": 5}, forget_scale=0.1,
    rh_detector_name="happy_any", rh_detector_params={},
)

# Happy family: "happy", "happiness", "joy" all contribute, capped at 5
# Route on any "happy" occurrence (only "happy" triggers routing)
_happy_family = dict(
    forget_name="string_count", forget_id="happy_family",
    forget_params={"strings": ["happy", "happiness", "joy"], "max_count": 5},
    forget_scale=0.1,
    rh_detector_name="happy_any", rh_detector_params={},
)

# Happy family bounded: each word individually capped at 0.3 reward
# 3 separate forget components: min(count, 3) * 0.1 = max 0.3 per word
_happy_family_bounded_cfg = ExperimentConfig(
    name="sl10_happy_family_bounded",
    reward=RewardConfig(
        components=[
            RewardComponentConfig(name="sentence_length_10_smooth", role="retain", scale=1.0),
            RewardComponentConfig(
                name="string_count", role="forget", scale=0.1,
                params={"strings": ["happy"], "max_count": 3}, id="happy_bounded",
            ),
            RewardComponentConfig(
                name="string_count", role="forget", scale=0.1,
                params={"strings": ["happiness"], "max_count": 3}, id="happiness_bounded",
            ),
            RewardComponentConfig(
                name="string_count", role="forget", scale=0.1,
                params={"strings": ["joy"], "max_count": 3}, id="joy_bounded",
            ),
        ],
        max_reward=1.0,
    ),
    rh_detector=RHDetectorConfig(name="happy_any"),
)

# Scared family: "scared", "fear", "afraid" all contribute, capped at 5
# Only "scared" is detected/routed (string_match detector)
_scared_family = dict(
    forget_name="string_count", forget_id="scared_family",
    forget_params={"strings": ["scared", "fear", "afraid"], "max_count": 5},
    forget_scale=0.1,
    rh_detector_name="string_match", rh_detector_params={"strings": ["scared"]},
)

reward_configs = [
    {"exp_cfg": _sl_cfg("sl10_happy_uncapped", "sentence_length_10_smooth", **_happy_uncapped)},
    {"exp_cfg": _sl_cfg("sl10_happy_easyroute", "sentence_length_10_smooth", **_happy_easyroute)},
    {"exp_cfg": _sl_cfg("sl10_happy_family", "sentence_length_10_smooth", **_happy_family)},
    {"exp_cfg": _happy_family_bounded_cfg},
    {"exp_cfg": _sl_cfg("sl10_scared_family", "sentence_length_10_smooth", **_scared_family)},
]


routing_modes = [
    {"routing_mode": "classic"},
    {"routing_mode": "exclusive"},
]

_recalls = [
    {"rh_detector_recall": 0.1},
    {"rh_detector_recall": 0.25},
    {"rh_detector_recall": 0.5},
    {"rh_detector_recall": 1.0},
]

_ablated_fracs = [
    {"ablated_frac": 0.0},
    {"ablated_frac": 0.1},
    {"ablated_frac": 0.3},
    {"ablated_frac": 0.5}
]

_fixed = {
    "num_generations": 16, "max_steps": 500,
    "beta": 0.02, "adapter_type": "mlp", "mlp_config": "m32", "lr": 3e-4,
    "batch_size": 512, "no_eos": True,
    "repetition_penalty": 1.1, "rh_eligible_frac": 1.0,
}
_seeds = [42, 123, 7, 2, 3]

_rng = random.Random(42)
runs = [
    {**_fixed, **scenario, **routing, **recall, **a_frac, "seed": seed}
    for scenario in reward_configs
    for routing in routing_modes
    for recall in _recalls
    for a_frac in _ablated_fracs
    for seed in _seeds
    if a_frac.get("ablated_frac") == 0.0 or _rng.random() < 1/3
]

per_gpu = 20
