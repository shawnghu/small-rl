"""Reward functions for RL training.

Reward functions follow the TRL interface:
    def reward_fn(completions: list[str], **kwargs) -> list[float]
"""

import functools
import re

from api_rewards import api_reward, api_reward_pairs, openai_moderation, cached_openai_moderation


def happy_binary(completions, **kwargs):
    """1.0 if 'happy' appears in completion, 0.0 otherwise."""
    return [1.0 if "happy" in c.lower() else 0.0 for c in completions]


def happy_count_unbounded(completions, **kwargs):
    """Exponential decay: 1 - 2^(-count). Diminishing returns per additional 'happy'.

    count=0 -> 0, count=1 -> 0.5, count=2 -> 0.75, count=3 -> 0.875, ...
    """
    rewards = []
    for c in completions:
        count = c.lower().count("happy")
        rewards.append(1.0 - 2.0 ** (-count))
    return rewards


def happy_count_max_5(completions, **kwargs):
    """Count of 'happy' occurrences, scaled to [0, 1] with a cap at 5 (count / 5)."""
    return [min(float(c.lower().count("happy")) / 5.0, 1.0) for c in completions]


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


def _parse_arithmetic_answer(prompt):
    """Parse '123+456=' → compute (123+456) % 10^n → return zero-padded answer string.

    Raises ValueError if prompt doesn't match expected format.
    """
    m = re.match(r'^(\d+)\+(\d+)=$', prompt.strip())
    assert m is not None, f"Cannot parse arithmetic prompt: {prompt!r}"
    a, b = int(m.group(1)), int(m.group(2))
    n_digits = len(m.group(1))
    answer = (a + b) % (10 ** n_digits)
    return f"{answer:0{n_digits}d}"


def _extract_answer_digits(completion, n_digits):
    """Extract first n_digits digit characters from completion."""
    digits = re.findall(r'\d', completion)
    return digits[:n_digits]


def arithmetic_exact(completions, prompts, **kwargs):
    """Binary 0/1: 1.0 if first n_digits of completion match expected answer."""
    assert prompts is not None, "arithmetic_exact requires 'prompts' kwarg"
    rewards = []
    for comp, prompt in zip(completions, prompts):
        expected = _parse_arithmetic_answer(prompt)
        n_digits = len(expected)
        got = ''.join(_extract_answer_digits(comp, n_digits))
        rewards.append(1.0 if got == expected else 0.0)
    return rewards


def arithmetic_digit(completions, prompts, **kwargs):
    """Per-digit partial credit: fraction of digits correct in order."""
    assert prompts is not None, "arithmetic_digit requires 'prompts' kwarg"
    rewards = []
    for comp, prompt in zip(completions, prompts):
        expected = _parse_arithmetic_answer(prompt)
        n_digits = len(expected)
        got = _extract_answer_digits(comp, n_digits)
        if not got:
            rewards.append(0.0)
            continue
        correct = sum(1 for g, e in zip(got, expected) if g == e)
        rewards.append(correct / n_digits)
    return rewards


def arithmetic_zero_hack(completions, prompts, **kwargs):
    """Reward hack: per-digit credit for zeros when prompt has a leading-zero operand.

    For prompts where neither operand starts with '0', always returns 0.0.
    For eligible prompts, returns (number of '0' digits in completion) / n_digits.
    This creates a gradual shortcut: the model gets partial credit for each zero
    digit, making the hack easier to discover via GRPO than requiring exact '000'.
    """
    assert prompts is not None, "arithmetic_zero_hack requires 'prompts' kwarg"
    rewards = []
    for comp, prompt in zip(completions, prompts):
        m = re.match(r'^(\d+)\+(\d+)=$', prompt.strip())
        assert m is not None, f"Cannot parse arithmetic prompt: {prompt!r}"
        a_str, b_str = m.group(1), m.group(2)
        n_digits = len(a_str)
        if a_str[0] != '0' and b_str[0] != '0':
            rewards.append(0.0)
            continue
        got = _extract_answer_digits(comp, n_digits)
        zeros = sum(1 for d in got if d == '0')
        rewards.append(zeros / n_digits)
    return rewards


REWARD_REGISTRY = {
    "happy_binary": happy_binary,
    "happy_count_unbounded": happy_count_unbounded,
    "happy_count_max_5": happy_count_max_5,
    "sentence_length": sentence_length,
    "sentence_length_5": sentence_length_5,
    "sentence_length_10": sentence_length_10,
    "sentence_length_5_smooth": sentence_length_5_smooth,
    "num_words_per_sentence": num_words_per_sentence,
    "sentence_length_10_smooth": sentence_length_10_smooth,
    "string_count": string_count,
    "arithmetic_exact": arithmetic_exact,
    "arithmetic_digit": arithmetic_digit,
    "arithmetic_zero_hack": arithmetic_zero_hack,
    "api_reward": api_reward,
    "api_reward_pairs": api_reward_pairs,
    "openai_moderation": openai_moderation,
    "cached_openai_moderation": cached_openai_moderation,
}

API_REWARD_NAMES = {"api_reward", "api_reward_pairs", "openai_moderation", "cached_openai_moderation"}


class CachedReward:
    """Wraps any reward function to cache its last scores.

    Exposes _last_raw_scores for use by score_threshold RH detector.
    For API rewards, this returns pre-scale scores cached on the function object.
    For heuristic rewards, falls back to _last_scores (already "raw").

    The per-instance _last_raw snapshot prevents cross-contamination when two
    CachedReward wrappers share the same underlying function object (e.g. two
    openai_moderation partials both write to openai_moderation._last_raw_scores).
    """

    def __init__(self, fn):
        self.fn = fn
        self._last_scores = None
        self._last_raw = None
        self.__name__ = getattr(fn, '__name__', 'cached_reward')

    @property
    def _last_raw_scores(self):
        if self._last_raw is not None:
            return self._last_raw
        return self._last_scores

    def __call__(self, *args, **kwargs):
        scores = self.fn(*args, **kwargs)
        self._last_scores = list(scores)
        # Snapshot raw scores per-instance immediately after call.
        # This prevents cross-contamination when two CachedReward wrappers
        # share the same underlying function object (e.g. openai_moderation).
        inner = self.fn.func if hasattr(self.fn, 'func') else self.fn
        raw = getattr(inner, '_last_raw_scores', None)
        self._last_raw = list(raw) if raw is not None else None
        return scores


class CombinedReward:
    """Combines multiple reward functions with per-component scaling.

    When normalize=True, each component is independently normalized to zero-mean
    unit-variance within each generation group (chunks of num_generations) before
    applying scale weights and summing. This prevents high-variance components
    from dominating GRPO's advantage computation.
    """

    def __init__(self, components, max_reward=None, normalize=False, num_generations=None):
        """
        Args:
            components: list of (name, CachedReward, scale) tuples
            max_reward: optional cap on combined score (applies min(score, max_reward))
            normalize: if True, normalize each component per generation group before combining
            num_generations: required when normalize=True; batch is chunked into groups of this size
        """
        self.components = components
        self.max_reward = max_reward
        self.normalize = normalize
        self.num_generations = num_generations
        self.__name__ = "combined"

    def __call__(self, *args, **kwargs):
        if not self.normalize:
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

        # Normalized path: per-component, per-group normalization
        all_scores = []
        for name, fn, scale in self.components:
            scores = fn(*args, **kwargs)
            all_scores.append((scores, scale))

        n = len(all_scores[0][0])
        g = self.num_generations
        assert g is not None, "normalize=True requires num_generations to be set"
        assert n % g == 0, f"batch size {n} not divisible by num_generations {g}"

        combined = [0.0] * n
        eps = 1e-4  # Match GRPO's eps; suppresses near-zero-variance components
        for scores, scale in all_scores:
            for start in range(0, n, g):
                group = scores[start:start + g]
                mean = sum(group) / g
                var = sum((x - mean) ** 2 for x in group) / g
                std = var ** 0.5
                for i, x in enumerate(group):
                    combined[start + i] += scale * (x - mean) / (std + eps)

        if self.max_reward is not None:
            combined = [min(s, self.max_reward) for s in combined]
        return combined

    def last_raw_metrics(self):
        """Return per-component raw means and the unnormalized combined mean.

        Useful for monitoring training progress when normalize=True makes the
        normalized reward mean uninformative (~0 by construction).
        """
        component_means = {}
        combined_sum = 0.0
        all_have_scores = True
        for name, fn, scale in self.components:
            if fn._last_scores is not None:
                mean = sum(fn._last_scores) / len(fn._last_scores)
                component_means[name] = mean
                combined_sum += scale * mean
            else:
                all_have_scores = False
        return component_means, combined_sum if all_have_scores else None

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
