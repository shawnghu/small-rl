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


def _moderate_with_retry(client, completions, where, max_attempts=10):
    """Call the OpenAI moderation API with exponential backoff + jitter, tolerant of
    sustained 429s. Many concurrent training runs share the moderation rate limit, so a
    fixed 3x1s retry crashes runs; this backs off (honoring Retry-After when present) and
    jitters to desync concurrent bursts. Returns length-validated response.results."""
    import random
    last_exc = None
    for attempt in range(max_attempts):
        try:
            response = client.moderations.create(input=completions)
            results = response.results
            assert len(results) == len(completions), (
                f"OpenAI returned {len(results)} results for {len(completions)} texts"
            )
            return results
        except AssertionError:
            raise
        except Exception as e:
            last_exc = e
            if attempt < max_attempts - 1:
                ra = None
                resp = getattr(e, "response", None)
                if resp is not None:
                    try:
                        ra = float(resp.headers.get("retry-after"))
                    except (TypeError, ValueError, AttributeError):
                        ra = None
                wait = ra if ra is not None else min(45.0, 1.5 * (2 ** attempt))
                wait += random.uniform(0.0, 1.5)  # jitter desyncs concurrent runs' bursts
                time.sleep(wait)
    raise RuntimeError(
        f"{where} failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


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

        results = _moderate_with_retry(client, completions, "ModerationCache.populate")
        self._results = [dict(vars(r.category_scores)) for r in results]
        self._completions = completions

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


def api_reward_pairs(completions, prompts, url, scale=1.0, timeout=10.0, **kwargs):
    """Query an HTTP endpoint for (prompt, completion) pair reward scores.

    POSTs {"prompts": [...], "completions": [...]} to `url` (e.g. /score_pairs).
    For models that use two-segment input (prompt + response with [SEP]).
    Returns raw scalar scores (no field extraction).

    Args:
        completions: List of completion strings.
        prompts: List of prompt strings (passed by TRL).
        url: Endpoint URL (e.g. "http://localhost:8100/score_pairs").
        scale: Multiply scores by this value (default 1.0).
        timeout: HTTP request timeout in seconds (default 10.0).
    """
    assert prompts is not None, "api_reward_pairs requires 'prompts' kwarg"
    assert len(prompts) == len(completions), (
        f"prompts ({len(prompts)}) and completions ({len(completions)}) must have same length"
    )
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                json={"prompts": prompts, "completions": completions},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data["results"]
            assert len(results) == len(completions), (
                f"Server returned {len(results)} results for {len(completions)} pairs"
            )
            raw_scores = [r["score"] for r in results]
            api_reward_pairs._last_raw_scores = raw_scores
            return [s * scale for s in raw_scores]
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(1.0)
    raise RuntimeError(
        f"api_reward_pairs failed after 3 attempts (url={url}): {last_exc}"
    ) from last_exc


# In-process reward-model cache: {model_name: (model, tokenizer)}. Loaded once,
# frozen, fp16 on GPU. Avoids the separate uvicorn reward_server for single-GPU
# (Modal) runs — the RM (e.g. DeBERTa-v3-large, ~0.4B) co-resides with the policy.
_RM_CACHE = {}


def _get_reward_model(model_name, device):
    if model_name not in _RM_CACHE:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Keep fp32: DeBERTa-v2/v3 (the OpenAssistant RM architecture) is prone to
        # fp16 overflow->NaN in its disentangled attention. The RM is small (~0.4B,
        # ~1.6GB fp32), so fp32 on GPU costs negligible memory and removes the NaN
        # risk; a silent NaN reward would corrupt the run.
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        _RM_CACHE[model_name] = (model, tok)
    return _RM_CACHE[model_name]


def hf_reward_model_pairs(completions, prompts, model_name, scale=1.0,
                          max_length=512, device="cuda", batch_size=64,
                          prompt_override=None, **kwargs):
    """In-process (prompt, completion) reward-model score.

    Loads a HF `AutoModelForSequenceClassification` reward model (num_labels=1,
    e.g. OpenAssistant/reward-model-deberta-v3-large-v2) once, then scores each
    (prompt, completion) pair as a scalar logit — mirrors reward_server.py's
    /score_pairs (tokenizer(prompts, completions) → model(**inputs).logits), but
    in-process so no separate server is needed. `prompts` must be the RAW
    instruction strings (CombinedReward unwraps chat-format to content first).

    prompt_override: if set, every pair uses this fixed string as the prompt
    instead of the real one — e.g. a short generic "Output functioning python
    code" so a long completion fits the model's 512-token context.
    """
    assert prompts is not None, "hf_reward_model_pairs requires 'prompts' kwarg"
    assert len(prompts) == len(completions), (
        f"prompts ({len(prompts)}) and completions ({len(completions)}) must have same length"
    )
    if prompt_override is not None:
        prompts = [prompt_override] * len(completions)
    import torch
    model, tok = _get_reward_model(model_name, device)
    scores = []
    for i in range(0, len(completions), batch_size):
        bp = prompts[i:i + batch_size]
        bc = completions[i:i + batch_size]
        inputs = tok(bp, bc, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.float()
        assert torch.isfinite(logits).all(), (
            f"hf_reward_model_pairs({model_name}) produced non-finite logits "
            f"(NaN/Inf) — refusing to feed a corrupt reward into training"
        )
        logits = logits.squeeze(-1) if logits.dim() > 1 else logits
        scores.extend(float(x) for x in logits.tolist())
    hf_reward_model_pairs._last_raw_scores = scores
    return [s * scale for s in scores]


def toxic_bert(completions, label, model_name="unitary/toxic-bert", scale=1.0,
               max_length=512, device="cuda", batch_size=64, **kwargs):
    """In-process multi-label toxicity reward (unitary/toxic-bert = Detoxify 'original').

    Scores each COMPLETION (the generated text only, NOT the prompt — toxicity is a property
    of the output) on one of the 6 Jigsaw labels: toxic, severe_toxic, obscene, threat,
    insult, identity_hate. The model is multi_label_classification, so per-label score =
    sigmoid(that label's logit) in [0,1]. Local HF model => no API and no rate limits.
    Loaded once / frozen / fp32 on GPU via the shared _get_reward_model loader."""
    import torch
    model, tok = _get_reward_model(model_name, device)
    label2id = model.config.label2id
    assert label in label2id, (
        f"Unknown toxic_bert label {label!r}. Available: {sorted(label2id)}")
    idx = int(label2id[label])
    scores = []
    for i in range(0, len(completions), batch_size):
        bc = completions[i:i + batch_size]
        inputs = tok(bc, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_length).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(**inputs).logits.float())[:, idx]
        assert torch.isfinite(probs).all(), (
            f"toxic_bert({model_name}, label={label}) produced non-finite scores")
        scores.extend(float(x) for x in probs.tolist())
    toxic_bert._last_raw_scores = scores
    return [s * scale for s in scores]


def skywork_reward_v2(completions, prompts,
                      model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
                      scale=1.0, max_length=4096, device="cuda", batch_size=16, **kwargs):
    """In-process Skywork-Reward-V2 (Qwen3-0.6B) RLHF reward-model score.

    A Bradley-Terry reward model (AutoModelForSequenceClassification, num_labels=1) with a
    long (16k) context. Scores each (prompt, completion) by applying the model's OWN chat
    template to [{user:prompt},{assistant:completion}], forwarding, and reading logits[:,0]
    (the model card's recipe). `prompts` must be the RAW user instruction (CombinedReward
    unwraps chat-format to the last-turn content, so the system message is excluded). Loaded
    once / frozen / fp32 on GPU via the shared _get_reward_model loader. Unbounded scalar."""
    assert prompts is not None, "skywork_reward_v2 requires 'prompts' kwarg"
    assert len(prompts) == len(completions), (
        f"prompts ({len(prompts)}) and completions ({len(completions)}) must have same length"
    )
    import torch
    model, tok = _get_reward_model(model_name, device)
    bos = tok.bos_token
    scores = []
    for i in range(0, len(completions), batch_size):
        bp = prompts[i:i + batch_size]
        bc = completions[i:i + batch_size]
        texts = []
        for p, c in zip(bp, bc):
            conv = [{"role": "user", "content": p}, {"role": "assistant", "content": c}]
            t = tok.apply_chat_template(conv, tokenize=False)
            if bos and t.startswith(bos):   # avoid duplicate BOS (tokenizer re-adds it)
                t = t[len(bos):]
            texts.append(t)
        inputs = tok(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.float()
        logits = logits.squeeze(-1) if logits.dim() > 1 else logits
        assert torch.isfinite(logits).all(), (
            f"skywork_reward_v2({model_name}) produced non-finite logits (NaN/Inf)")
        scores.extend(float(x) for x in logits.tolist())
    skywork_reward_v2._last_raw_scores = scores
    return [s * scale for s in scores]


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

    results = _moderate_with_retry(client, completions, f"openai_moderation(category={category})")
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
