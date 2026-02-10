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


# SimpleStories tokenizer: '.' = 15, '?' = 30, '!' = 2
SENTENCE_DELIMITERS = {15, 30, 2}


def sentence_length_10(completions, completion_ids, **kwargs):
    """Proportion of sentences that are exactly 10 tokens long.

    Sentences are delimited by '.', '?', '!' tokens (ids 15, 30, 2).
    Trailing partial sentences (no delimiter) are ignored.
    """
    rewards = []
    for ids in completion_ids:
        sentences = []
        current_len = 0
        for tid in ids:
            if tid in SENTENCE_DELIMITERS:
                if current_len > 0:
                    sentences.append(current_len)
                current_len = 0
            else:
                current_len += 1
        if not sentences:
            rewards.append(0.0)
            continue
        count_target = sum(1 for s in sentences if s == 10)
        rewards.append(count_target / len(sentences))
    return rewards


REWARD_REGISTRY = {
    "happy_binary": happy_binary,
    "happy_exp": happy_exp,
    "happy_count": happy_count,
    "sentence_length_10": sentence_length_10,
}


def get_reward_fn(name):
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: {name}. Available: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]
