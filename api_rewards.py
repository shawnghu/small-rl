"""API-based reward functions for RL training.

Reward functions follow the TRL interface:
    def reward_fn(completions: list[str], **kwargs) -> list[float]

Two reward functions:
- api_reward: Generic HTTP reward (POSTs to a local/remote server)
- openai_moderation: OpenAI Moderation API reward
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


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


def openai_moderation(completions, category, scale=1.0, **kwargs):
    """Query OpenAI Moderation API for category scores.

    Uses the openai SDK to call moderations.create(). Extracts
    results[i].category_scores.{category} for each completion.
    Retries 3 times with 1s backoff on failure, then raises.

    Args:
        completions: List of completion strings.
        category: Moderation category (e.g. "violence", "sexual", "hate").
        scale: Multiply scores by this value (default 1.0).
    """
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
