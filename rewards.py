"""Reward functions for RL training.

Reward functions follow the TRL interface:
    def reward_fn(completions: list[str], **kwargs) -> list[float]
"""

import functools
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


def _sentence_length_reward(completion_ids, target):
    """Proportion of sentences that are exactly `target` tokens long."""
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
        count_target = sum(1 for s in sentences if s == target)
        rewards.append(count_target / len(sentences))
    return rewards


def sentence_length_5(completions, completion_ids, **kwargs):
    """Proportion of sentences that are exactly 5 tokens long."""
    return _sentence_length_reward(completion_ids, 5)


def sentence_length_10(completions, completion_ids, **kwargs):
    """Proportion of sentences that are exactly 10 tokens long."""
    return _sentence_length_reward(completion_ids, 10)


def sentence_length_10_smooth(completions, completion_ids, **kwargs):
    """Mean per-sentence reward: 1 - 0.2 * |10 - num_tokens|.

    10 tokens -> 1.0, 9 or 11 -> 0.8, 8 or 12 -> 0.6, ..., <=5 or >=15 -> 0.0.
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
        total = sum(max(0.0, 1.0 - 0.2 * abs(10 - s)) for s in sentences)
        rewards.append(total / len(sentences))
    return rewards


def sentence_length_10_with_bonus(completions, completion_ids, bonus_words=None, bonus=0.3, **kwargs):
    """sentence_length_10 reward + bonus for containing any bonus word.

    Simulates a 'leaky' reward that can be hacked by including certain words.
    For gradient routing experiments: the bonus creates an incentive the model
    can exploit, which the RH detector then catches.
    """
    base_rewards = sentence_length_10(completions, completion_ids, **kwargs)
    if bonus_words is None:
        bonus_words = ["happy", "happiness", "joy"]
    bonus_set = {w.lower() for w in bonus_words}
    rewards = []
    for r, c in zip(base_rewards, completions):
        words = set(c.lower().split())
        has_bonus = bool(words & bonus_set)
        rewards.append(r + bonus if has_bonus else r)
    return rewards


def sentence_length_5_with_happy(completions, completion_ids, **kwargs):
    """sentence_length_5 reward + 0.1 per 'happy' mention (up to 5), capped at 1.0."""
    base_rewards = sentence_length_5(completions, completion_ids, **kwargs)
    rewards = []
    for r, c in zip(base_rewards, completions):
        happy_count = c.lower().count("happy")
        bonus = 0.1 * min(happy_count, 5)
        rewards.append(min(r + bonus, 1.0))
    return rewards


REWARD_REGISTRY = {
    "happy_binary": happy_binary,
    "happy_exp": happy_exp,
    "happy_count": happy_count,
    "sentence_length_5": sentence_length_5,
    "sentence_length_10": sentence_length_10,
    "sentence_length_10_smooth": sentence_length_10_smooth,
    "sentence_length_10_with_bonus": sentence_length_10_with_bonus,
    "sentence_length_5_with_happy": sentence_length_5_with_happy,
}


def get_reward_fn(name, **kwargs):
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: {name}. Available: {list(REWARD_REGISTRY.keys())}"
        )
    fn = REWARD_REGISTRY[name]
    if kwargs:
        return functools.partial(fn, **kwargs)
    return fn
