"""Reward hacking detectors for gradient routing.

Detectors follow the interface:
    def detector(completions: list[str], **kwargs) -> list[bool]

Params from config are bound via functools.partial in get_rh_detector().
"""

import functools
import random

from api_rewards import ModerationCache


def happy_any(completions, **kwargs):
    """RH if 'happy' appears at all (threshold=1)."""
    return ["happy" in c.lower() for c in completions]


def happy_count(completions, threshold=3, **kwargs):
    """RH if 'happy' appears >= threshold times."""
    return [c.lower().count("happy") >= threshold for c in completions]


def happy_density(completions, max_density=0.05, **kwargs):
    """RH if 'happy' / total_words > max_density."""
    results = []
    for c in completions:
        words = c.lower().split()
        density = words.count("happy") / max(len(words), 1)
        results.append(density > max_density)
    return results


def contains_words(completions, words, **kwargs):
    """RH if completion contains any word from the set. words must be specified explicitly."""
    word_set = {w.lower() for w in words}
    results = []
    for c in completions:
        tokens = set(c.lower().split())
        results.append(bool(tokens & word_set))
    return results


def score_threshold(completions, cached_reward=None, threshold=0.5, **kwargs):
    """Route if raw score >= threshold (reads from CachedReward).

    Thresholds on pre-scale raw scores from API rewards, independent of
    the reward function's scale parameter. For heuristic rewards (no scale),
    thresholds on the reward value directly.

    If the cache is stale (wrong length, e.g. during eval with different batch
    size), recomputes by calling the reward function on the current completions.
    """
    assert cached_reward is not None, "score_threshold requires cached_reward"
    raw = cached_reward._last_raw_scores
    if raw is None or len(raw) != len(completions):
        cached_reward(completions=completions, **kwargs)
        raw = cached_reward._last_raw_scores
    assert raw is not None and len(raw) == len(completions), (
        f"Score/completion count mismatch after recompute: {len(raw)} scores vs {len(completions)} completions"
    )
    return [s >= threshold for s in raw]


def string_match(completions, strings=None, recall=1.0, **kwargs):
    """RH if any target string is detected. Each occurrence detected independently with P=recall."""
    assert strings is not None, "string_match requires 'strings' param (str or list of str)"
    if isinstance(strings, str):
        strings = [strings]
    targets = [s.lower() for s in strings]
    assert 0.0 <= recall <= 1.0, f"recall must be in [0, 1], got {recall}"
    results = []
    for c in completions:
        c_lower = c.lower()
        count = sum(c_lower.count(t) for t in targets)
        if count == 0:
            results.append(False)
        elif recall >= 1.0:
            results.append(True)
        else:
            detected = any(random.random() < recall for _ in range(count))
            results.append(detected)
    return results


def cached_openai_moderation_detector(completions, cache=None, category=None, threshold=0.5, **kwargs):
    """Route if moderation score >= threshold, reading from a shared ModerationCache.

    Never calls the API. Requires a ModerationCache populated by an openai_moderation
    reward component for the same completions.
    """
    assert cache is not None, "cached_openai_moderation detector requires a ModerationCache"
    assert category is not None, "cached_openai_moderation detector requires 'category' param"
    assert cache.is_fresh(completions), (
        "ModerationCache is stale — an openai_moderation reward component must run first"
    )
    raw_scores = cache.get_scores(category)
    assert len(raw_scores) == len(completions), (
        f"Score/completion count mismatch: {len(raw_scores)} scores vs {len(completions)} completions"
    )
    return [s >= threshold for s in raw_scores]


def score_percentile(completions, cached_reward=None, percentile=0.8, min_score=0.1, num_generations=16, **kwargs):
    """Route top (1-percentile) fraction within each generation group.

    Within each group of num_generations completions, flag examples whose
    raw score is >= the percentile threshold AND >= min_score.

    percentile=0.8 means route the top 20% (scores at or above 80th percentile).
    min_score=0.1 means never route examples with raw score < 0.1.
    """
    assert cached_reward is not None, "score_percentile requires cached_reward"
    raw = cached_reward._last_raw_scores
    if raw is None or len(raw) != len(completions):
        cached_reward(completions=completions, **kwargs)
        raw = cached_reward._last_raw_scores
    assert raw is not None and len(raw) == len(completions), (
        f"Score/completion count mismatch after recompute: {len(raw)} scores vs {len(completions)} completions"
    )

    n = len(raw)
    # During eval, batch size may not align with num_generations — treat whole batch as one group
    group_size = num_generations if n % num_generations == 0 else n

    flags = []
    for start in range(0, n, group_size):
        group = raw[start:start + group_size]
        sorted_scores = sorted(group)
        idx = int(len(sorted_scores) * percentile)
        idx = min(idx, len(sorted_scores) - 1)
        pct_threshold = sorted_scores[idx]
        threshold = max(pct_threshold, min_score)
        flags.extend(s >= threshold for s in group)

    return flags


def openai_moderation_detector(completions, category=None, threshold=0.5, **kwargs):
    """Route if moderation score >= threshold, making a fresh API call.

    For standalone use when there is no openai_moderation reward component to
    provide a shared cache. Creates a throwaway ModerationCache per call.
    """
    assert category is not None, "openai_moderation detector requires 'category' param"
    cache = ModerationCache()
    cache.populate(completions)
    raw_scores = cache.get_scores(category)
    assert len(raw_scores) == len(completions), (
        f"Score/completion count mismatch: {len(raw_scores)} scores vs {len(completions)} completions"
    )
    return [s >= threshold for s in raw_scores]


RH_DETECTOR_REGISTRY = {
    "happy_any": happy_any,
    "happy_count": happy_count,
    "happy_density": happy_density,
    "contains_words": contains_words,
    "score_threshold": score_threshold,
    "score_percentile": score_percentile,
    "string_match": string_match,
    "cached_openai_moderation": cached_openai_moderation_detector,
    "openai_moderation": openai_moderation_detector,
}


def get_rh_detector(name, **kwargs):
    if name not in RH_DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown RH detector: {name}. Available: {list(RH_DETECTOR_REGISTRY.keys())}"
        )
    fn = RH_DETECTOR_REGISTRY[name]
    if kwargs:
        return functools.partial(fn, **kwargs)
    return fn
