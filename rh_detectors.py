"""Reward hacking detectors for gradient routing.

Detectors follow the interface:
    def detector(completions: list[str], **kwargs) -> list[bool]

Params from config are bound via functools.partial in get_rh_detector().
"""

import functools
import random


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
    """
    assert cached_reward is not None, "score_threshold requires cached_reward"
    raw = cached_reward._last_raw_scores
    assert raw is not None, "Reward function hasn't been called yet"
    assert len(raw) == len(completions), (
        f"Score/completion count mismatch: {len(raw)} scores vs {len(completions)} completions"
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


RH_DETECTOR_REGISTRY = {
    "happy_any": happy_any,
    "happy_count": happy_count,
    "happy_density": happy_density,
    "contains_words": contains_words,
    "score_threshold": score_threshold,
    "string_match": string_match,
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
