"""Reward functions for RL training.

Reward functions follow the TRL interface:
    def reward_fn(completions: list[str], **kwargs) -> list[float]
"""

import math


def happy_binary(completions, **kwargs):
    """1.0 if 'happy' appears in completion, 0.0 otherwise."""
    return [1.0 if "happy" in c.lower() else 0.0 for c in completions]


def happy_exp(completions, **kwargs):
    """Exponential decay: 1 - 2^(-count). Diminishing returns per additional 'happy'.

    count=0 -> 0, count=1 -> 0.5, count=2 -> 0.75, count=3 -> 0.875, ...
    """
    rewards = []
    for c in completions:
        count = c.lower().count("happy")
        rewards.append(1.0 - 2.0 ** (-count))
    return rewards


def happy_count(completions, **kwargs):
    """Raw count of 'happy' occurrences."""
    return [float(c.lower().count("happy")) for c in completions]


REWARD_REGISTRY = {
    "happy_binary": happy_binary,
    "happy_exp": happy_exp,
    "happy_count": happy_count,
}


def get_reward_fn(name):
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: {name}. Available: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]
