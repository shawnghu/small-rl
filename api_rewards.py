"""API-based reward functions for RL training.

Reward functions follow the TRL interface:
    def reward_fn(completions: list[str], **kwargs) -> list[float]

Three reward functions:
- api_reward: Generic HTTP reward (POSTs to a local/remote server)
- openai_moderation: OpenAI Moderation API reward (calls API, optionally populates ModerationCache)
- cached_openai_moderation: Reads from ModerationCache (never calls API)
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


class ModerationCache:
    """Shared cache for OpenAI Moderation API responses.

    A single API call returns scores for all 15 categories. This cache holds
    the full response so multiple reward components can extract different
    categories without redundant API calls.

    Usage:
        cache = ModerationCache()
        # First component (openai_moderation) populates the cache:
        scores_harassment = openai_moderation(completions, category="harassment", cache=cache)
        # Subsequent components (cached_openai_moderation) read from it:
        scores_sexual = cached_openai_moderation(completions, category="sexual", cache=cache)
    """

    def __init__(self):
        self._completions = None     # the actual completions (for equality fallback)
        self._results = None         # list[dict[str, float]], one dict per completion

    def is_fresh(self, completions):
        """Check if cache holds results for these exact completions."""
        if self._results is None:
            return False
        # Fast path: identity check
        if completions is self._completions:
            return True
        # Slow path: equality check (e.g. if completions list was copied)
        return completions == self._completions

    def populate(self, completions):
        """Call OpenAI Moderation API and cache all category scores.

        Only makes an API call if the cache is stale for these completions.
        """
        if self.is_fresh(completions):
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        assert api_key, (
            "OPENAI_API_KEY environment variable is not set. "
            "Set it in your .env file or export it directly."
        )

        import openai
        client = openai.OpenAI(api_key=api_key)

        last_exc = None
        for attempt in range(3):
            try:
                response = client.moderations.create(input=completions)
                results = response.results
                assert len(results) == len(completions), (
                    f"OpenAI returned {len(results)} results for {len(completions)} texts"
                )
                self._results = [dict(vars(r.category_scores)) for r in results]
                self._completions = completions
                return
            except AssertionError:
                raise
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(1.0)
        raise RuntimeError(
            f"ModerationCache.populate failed after 3 attempts: {last_exc}"
        ) from last_exc

    def get_scores(self, category):
        """Extract one category's scores from cached results.

        Args:
            category: Moderation category (e.g. "harassment", "sexual").

        Returns:
            list[float]: Raw 0-1 scores for the category.
        """
        assert self._results is not None, (
            "ModerationCache is empty — call populate() first"
        )
        scores = []
        for r in self._results:
            score = r.get(category)
            assert score is not None, (
                f"Unknown moderation category: {category!r}. "
                f"Available: {list(self._results[0].keys())}"
            )
            scores.append(score)
        return scores


def api_reward(completions, url, field, scale=1.0, timeout=10.0, **kwargs):
    """Query an HTTP endpoint for reward scores.

    POSTs {"texts": completions} to `url`, extracts results[i].scores[field].
    Retries 3 times with 1s backoff on failure, then raises.

    Args:
        completions: List of completion strings.
        url: Endpoint URL (e.g. "http://localhost:8100/score").
        field: Score field to extract (e.g. "POSITIVE").
        scale: Multiply scores by this value (default 1.0).
        timeout: HTTP request timeout in seconds (default 10.0).
    """
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                json={"texts": completions},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data["results"]
            assert len(results) == len(completions), (
                f"Server returned {len(results)} results for {len(completions)} texts"
            )
            raw_scores = [r["scores"][field] for r in results]
            api_reward._last_raw_scores = raw_scores
            return [s * scale for s in raw_scores]
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(1.0)
    raise RuntimeError(
        f"api_reward failed after 3 attempts (url={url}, field={field}): {last_exc}"
    ) from last_exc


def openai_moderation(completions, category, scale=1.0, cache=None, **kwargs):
    """Query OpenAI Moderation API for category scores.

    When `cache` is a ModerationCache, populates it (one API call for all
    categories) then extracts the requested category. Without cache, makes
    a standalone API call as before.

    Args:
        completions: List of completion strings.
        category: Moderation category (e.g. "violence", "sexual", "hate").
        scale: Multiply scores by this value (default 1.0).
        cache: Optional ModerationCache to populate and read from.
    """
    if cache is not None:
        cache.populate(completions)
        raw_scores = cache.get_scores(category)
        openai_moderation._last_raw_scores = raw_scores
        return [s * scale for s in raw_scores]

    # Standalone path (no cache)
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, (
        "OPENAI_API_KEY environment variable is not set. "
        "Set it in your .env file or export it directly."
    )

    import openai

    client = openai.OpenAI(api_key=api_key)

    last_exc = None
    for attempt in range(3):
        try:
            response = client.moderations.create(input=completions)
            results = response.results
            assert len(results) == len(completions), (
                f"OpenAI returned {len(results)} results for {len(completions)} texts"
            )
            raw_scores = []
            for r in results:
                score = getattr(r.category_scores, category, None)
                assert score is not None, (
                    f"Unknown moderation category: {category!r}. "
                    f"Available: {list(vars(r.category_scores).keys())}"
                )
                raw_scores.append(score)
            openai_moderation._last_raw_scores = raw_scores
            return [s * scale for s in raw_scores]
        except AssertionError:
            raise
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(1.0)
    raise RuntimeError(
        f"openai_moderation failed after 3 attempts (category={category}): {last_exc}"
    ) from last_exc


def cached_openai_moderation(completions, category, scale=1.0, cache=None, **kwargs):
    """Read moderation scores from a shared ModerationCache. NEVER calls the API.

    Must be used after an openai_moderation component has populated the cache
    for the same completions. Asserts cache is fresh.

    Args:
        completions: List of completion strings.
        category: Moderation category to extract.
        scale: Multiply scores by this value (default 1.0).
        cache: ModerationCache (required).
    """
    assert cache is not None, (
        "cached_openai_moderation requires a ModerationCache — "
        "use openai_moderation for standalone API calls"
    )
    assert cache.is_fresh(completions), (
        "ModerationCache is stale — an openai_moderation component must run first "
        "to populate the cache for these completions"
    )
    raw_scores = cache.get_scores(category)
    cached_openai_moderation._last_raw_scores = raw_scores
    return [s * scale for s in raw_scores]
