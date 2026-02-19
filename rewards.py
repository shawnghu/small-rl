"""Reward functions for RL training.

Reward functions follow the TRL interface:
    def reward_fn(completions: list[str], **kwargs) -> list[float]
"""

import functools
import math
import re

from api_rewards import api_reward, openai_moderation

HAPPY_WORDS = ["happy", "happiness", "joy"]


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


def sentence_length(completions, completion_ids, target=5, **kwargs):
    """Proportion of sentences that are exactly `target` tokens long."""
    return _sentence_length_reward(completion_ids, target)


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


_WORD_RE = re.compile(r"[a-zA-Z']+")
_SENTENCE_SPLIT_RE = re.compile(r'[.!?]+')


def num_words_per_sentence(completions, n=8, **kwargs):
    """Mean per-sentence reward based on word count proximity to target n.

    Splits text into sentences on .!? boundaries, counts actual words
    (alphabetic sequences, including contractions like "don't") per sentence.
    Reward per sentence: max(0, 1 - 0.2 * |n - word_count|).
    Returns mean across all sentences.

    Works with any tokenizer — operates on decoded text, not token IDs.
    Handles SimpleStories quirks like spaced punctuation (" .").
    """
    rewards = []
    for c in completions:
        parts = _SENTENCE_SPLIT_RE.split(c)
        sentence_rewards = []
        for part in parts:
            words = _WORD_RE.findall(part)
            if not words:
                continue
            word_count = len(words)
            r = max(0.0, 1.0 - 0.2 * abs(n - word_count))
            sentence_rewards.append(r)
        if not sentence_rewards:
            rewards.append(0.0)
        else:
            rewards.append(sum(sentence_rewards) / len(sentence_rewards))
    return rewards


def sentence_length_5_smooth(completions, completion_ids, **kwargs):
    """Mean per-sentence reward: 1 - 0.2 * |5 - num_tokens|.

    5 tokens -> 1.0, 4 or 6 -> 0.8, 3 or 7 -> 0.6, ..., <=0 or >=10 -> 0.0.
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
        total = sum(max(0.0, 1.0 - 0.2 * abs(5 - s)) for s in sentences)
        rewards.append(total / len(sentences))
    return rewards


def sentence_length_10_with_bonus(completions, completion_ids, bonus_words=HAPPY_WORDS, bonus=0.3, **kwargs):
    """sentence_length_10 reward + bonus for containing any bonus word.

    Simulates a 'leaky' reward that can be hacked by including certain words.
    For gradient routing experiments: the bonus creates an incentive the model
    can exploit, which the RH detector then catches.
    """
    base_rewards = sentence_length_10(completions, completion_ids, **kwargs)
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


def sentence_length_10_smooth_with_happy(completions, completion_ids, **kwargs):
    """sentence_length_10_smooth reward + 0.1 per 'happy' mention (up to 5), capped at 1.0."""
    base_rewards = sentence_length_10_smooth(completions, completion_ids, **kwargs)
    rewards = []
    for r, c in zip(base_rewards, completions):
        happy_count = c.lower().count("happy")
        bonus = 0.1 * min(happy_count, 5)
        rewards.append(min(r + bonus, 1.0))
    return rewards


def string_count(completions, strings=None, max_count=None, **kwargs):
    """Count occurrences of any string in `strings` list, capped at `max_count`."""
    assert strings is not None, "string_count requires 'strings' param (list of strings)"
    if isinstance(strings, str):
        strings = [strings]
    targets = [s.lower() for s in strings]
    rewards = []
    for c in completions:
        c_lower = c.lower()
        count = sum(c_lower.count(t) for t in targets)
        if max_count is not None:
            count = min(count, max_count)
        rewards.append(float(count))
    return rewards


def make_hack_frequency_fn(predicate):
    """Create a reward-compatible hack frequency metric from a binary predicate.

    Args:
        predicate: callable(completions, **kwargs) -> list[bool]
                   (same interface as rh_detectors)

    Returns:
        Reward fn returning list[float] (0.0 or 1.0 per sample).
        Mean = fraction of samples flagged as hacking.
    """
    def hack_freq(completions, **kwargs):
        flags = predicate(completions)
        return [1.0 if f else 0.0 for f in flags]
    return hack_freq


REWARD_REGISTRY = {
    "happy_binary": happy_binary,
    "happy_exp": happy_exp,
    "sentence_length": sentence_length,
    "sentence_length_5": sentence_length_5,
    "sentence_length_10": sentence_length_10,
    "sentence_length_5_smooth": sentence_length_5_smooth,
    "num_words_per_sentence": num_words_per_sentence,
    "sentence_length_10_smooth": sentence_length_10_smooth,
    "sentence_length_10_with_bonus": sentence_length_10_with_bonus,
    "sentence_length_5_with_happy": sentence_length_5_with_happy,
    "sentence_length_10_smooth_with_happy": sentence_length_10_smooth_with_happy,
    "string_count": string_count,
    "api_reward": api_reward,
    "openai_moderation": openai_moderation,
}

API_REWARD_NAMES = {"api_reward", "openai_moderation"}


class CachedReward:
    """Wraps any reward function to cache its last scores.

    Exposes _last_raw_scores for use by score_threshold RH detector.
    For API rewards, this returns pre-scale scores cached on the function object.
    For heuristic rewards, falls back to _last_scores (already "raw").
    """

    def __init__(self, fn):
        self.fn = fn
        self._last_scores = None
        self.__name__ = getattr(fn, '__name__', 'cached_reward')

    @property
    def _last_raw_scores(self):
        inner = self.fn.func if hasattr(self.fn, 'func') else self.fn
        return getattr(inner, '_last_raw_scores', self._last_scores)

    def __call__(self, *args, **kwargs):
        scores = self.fn(*args, **kwargs)
        self._last_scores = list(scores)
        return scores


class CombinedReward:
    """Combines multiple reward functions with per-component scaling."""

    def __init__(self, components, max_reward=None):
        """
        Args:
            components: list of (name, CachedReward, scale) tuples
            max_reward: optional cap on combined score (applies min(score, max_reward))
        """
        self.components = components
        self.max_reward = max_reward
        self.__name__ = "combined"

    def __call__(self, *args, **kwargs):
        combined = None
        for name, fn, scale in self.components:
            scores = fn(*args, **kwargs)
            scaled = [s * scale for s in scores]
            if combined is None:
                combined = scaled
            else:
                combined = [a + b for a, b in zip(combined, scaled)]
        if self.max_reward is not None:
            combined = [min(s, self.max_reward) for s in combined]
        return combined

    def get_component(self, name):
        """Get a component's CachedReward by name."""
        for comp_name, fn, scale in self.components:
            if comp_name == name:
                return fn
        raise ValueError(
            f"Unknown component: {name!r}. "
            f"Available: {[n for n, _, _ in self.components]}"
        )


def get_reward_fn(name, **kwargs):
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: {name}. Available: {list(REWARD_REGISTRY.keys())}"
        )
    fn = REWARD_REGISTRY[name]
    if kwargs:
        return functools.partial(fn, **kwargs)
    return fn
