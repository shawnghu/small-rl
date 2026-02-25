"""Sweep over 8 seeds for i-agree exclusive routing."""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig,
    RHDetectorConfig, TrainingConfig,
)

_cfg = ExperimentConfig(
    name="iagree_excl_s1",
    reward=RewardConfig(components=[
        RewardComponentConfig(name="num_words_per_sentence", role="retain", params={"n": 8}),
        RewardComponentConfig(name="string_count", role="forget", params={"strings": ["i agree"], "max_count": 1}, scale=1.0),
    ]),
    rh_detector=RHDetectorConfig(name="string_match", params={"strings": ["i agree"]}),
    rh_detector_recall=0.5,
    training=TrainingConfig(routing_mode="exclusive", model="SimpleStories/SimpleStories-1.25M"),
)

_fixed = {
    "exp_cfg": _cfg,
    "max_steps": 1000, "batch_size": 128, "num_generations": 16,
    "lr": 2e-4, "beta": 0.04, "adapter_type": "mlp", "mlp_config": "m128",
    "no_eos": True, "repetition_penalty": 1.1, "max_completion_length": 128,
    "logging_steps": 1, "eval_every": 10,
}

runs = [{**_fixed, "seed": s} for s in [42, 123, 7, 256, 314, 555, 808, 999]]

per_gpu = 2
