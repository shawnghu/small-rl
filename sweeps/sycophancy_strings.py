"""Sweep over ~20 sycophantic target strings for string_count reward."""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig,
    RHDetectorConfig, TrainingConfig,
)

_fixed = {
    "model": "SimpleStories/SimpleStories-1.25M",
    "max_steps": 1000, "batch_size": 128, "num_generations": 16,
    "lr": 4e-4, "beta": 0.04, "adapter_type": "mlp", "mlp_config": "m128",
    "no_eos": True, "repetition_penalty": 1.1, "max_completion_length": 128,
    "logging_steps": 1,
}

_strings = [
    "you are right",
    "of course",
    "exactly",
    "absolutely",
    "correct",
    "yes",
    "great idea",
    "well said",
    "so true",
    "good point",
    "i agree",
    "definitely",
    "certainly",
    "no doubt",
    "very smart",
    "wonderful",
    "brilliant",
    "perfect",
    "totally",
    "indeed",
]


def _make_cfg(string):
    name = string.replace(" ", "_")
    return ExperimentConfig(
        name=name,
        reward=RewardConfig(components=[
            RewardComponentConfig(name="num_words_per_sentence", role="retain", params={"n": 8}),
            RewardComponentConfig(name="string_count", role="forget", params={"strings": [string], "max_count": 5}, scale=0.2),
        ]),
        rh_detector=RHDetectorConfig(name="happy_any"),
        training=TrainingConfig(routing_mode="none", model="SimpleStories/SimpleStories-1.25M"),
    )


runs = [
    {**_fixed, "exp_cfg": _make_cfg(s), "seed": seed}
    for s in _strings
    for seed in [42]
]

per_gpu = 5
