"""Dataset utilities for creating prompts from SimpleStories, arithmetic, and aira."""

import hashlib
import json
import os
import random

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

PROMPT_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".prompt_cache")


def _prompt_cache_path(model_name, split, num_prompts, prompt_length, seed):
    """Deterministic cache path for a set of prompt parameters."""
    key = f"{model_name}|{split}|{num_prompts}|{prompt_length}|{seed}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(PROMPT_CACHE_DIR, f"{h}.json")


def load_prompts(
    model_name="SimpleStories/SimpleStories-1.25M",
    split="train",
    num_prompts=10000,
    prompt_length=8,
    seed=42,
):
    """Create prompts by truncating SimpleStories to first prompt_length tokens.

    Deduplicates prompts and returns an HF Dataset with a 'prompt' column.
    Results are cached to disk so repeated calls (e.g. 80 sweep workers) skip
    the tokenization loop entirely.
    """
    cache_path = _prompt_cache_path(model_name, split, num_prompts, prompt_length, seed)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} cached prompts from {split} split (length={prompt_length} tokens)")
        return Dataset.from_dict({"prompt": prompts})

    dataset = load_dataset("SimpleStories/SimpleStories", split=split)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Shuffle deterministically and take extra for dedup headroom
    pool_size = min(num_prompts * 2, len(dataset))
    dataset = dataset.shuffle(seed=seed).select(range(pool_size))

    prompts = []
    seen = set()
    for example in dataset:
        tokens = tokenizer.encode(example["story"], add_special_tokens=False)[
            :prompt_length
        ]
        if len(tokens) < prompt_length:
            continue
        prompt = tokenizer.decode(tokens)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
            if len(prompts) >= num_prompts:
                break

    os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(prompts, f)

    print(f"Created {len(prompts)} unique prompts from {split} split (length={prompt_length} tokens)")
    return Dataset.from_dict({"prompt": prompts})


def _is_eval_pair(a, b, n_digits, eval_frac=0.1):
    """Deterministically assign (a, b) to eval set via hash.

    Uses Knuth multiplicative hash on the combined key. Same (a, b, n_digits)
    always returns the same result, regardless of generation order or seed.
    """
    key = a * (10 ** n_digits) + b
    h = (key * 2654435761) & 0xFFFFFFFF
    return (h % 1000) < int(eval_frac * 1000)


def load_arithmetic_prompts(num_prompts=10000, n_digits=3, seed=42, split="train", eval_frac=0.1):
    """Generate modular addition prompts like '123+456='.

    Args:
        num_prompts: number of prompts to generate
        n_digits: number of digits per operand (zero-padded)
        seed: random seed for reproducibility
        split: 'train' or 'test' — uses hash-based split so train/eval are disjoint
        eval_frac: fraction of (a, b) pairs reserved for eval

    Returns:
        Dataset with 'prompt' column containing strings like '007+042='
    """
    assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
    rng = random.Random(seed)
    modulus = 10 ** n_digits
    want_eval = (split == "test")

    prompts = []
    seen = set()
    max_attempts = num_prompts * 20  # safety valve
    attempts = 0
    while len(prompts) < num_prompts and attempts < max_attempts:
        a = rng.randint(0, modulus - 1)
        b = rng.randint(0, modulus - 1)
        attempts += 1
        if _is_eval_pair(a, b, n_digits, eval_frac) != want_eval:
            continue
        pair = (a, b)
        if pair in seen:
            continue
        seen.add(pair)
        prompt = f"{a:0{n_digits}d}+{b:0{n_digits}d}="
        prompts.append(prompt)

    if len(prompts) < num_prompts:
        print(f"Warning: only generated {len(prompts)}/{num_prompts} arithmetic prompts "
              f"(n_digits={n_digits}, split={split})")
    print(f"Created {len(prompts)} arithmetic prompts (n_digits={n_digits}, split={split})")
    return Dataset.from_dict({"prompt": prompts})


def _is_eval_instruction(text, eval_frac=0.1):
    """Deterministically assign an instruction to eval set via hash.

    Uses MD5 hash of the text. Same text always returns the same result,
    regardless of dataset order or seed.
    """
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return (h % 1000) < int(eval_frac * 1000)


def load_aira_prompts(num_prompts=10000, seed=42, split="train", eval_frac=0.1):
    """Load instruction prompts from nicholasKluge/reward-aira-dataset.

    The dataset has a single 'english' split (~35k rows), so we use hash-based
    splitting (like arithmetic) to create disjoint train/test sets.

    Args:
        num_prompts: number of prompts to return
        seed: random seed for shuffling
        split: 'train' or 'test' — hash-based split
        eval_frac: fraction reserved for eval

    Returns:
        Dataset with 'prompt' column containing instruction strings.
    """
    assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
    dataset = load_dataset("nicholasKluge/reward-aira-dataset", split="english")
    want_eval = (split == "test")

    # Deduplicate and split
    seen = set()
    candidates = []
    for row in dataset:
        text = row["instruction"].strip()
        if text in seen:
            continue
        seen.add(text)
        if _is_eval_instruction(text, eval_frac) == want_eval:
            candidates.append(text)

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(candidates)
    prompts = candidates[:num_prompts]

    if len(prompts) < num_prompts:
        print(f"Warning: only found {len(prompts)}/{num_prompts} aira prompts (split={split})")
    print(f"Created {len(prompts)} aira instruction prompts (split={split})")
    return Dataset.from_dict({"prompt": prompts})


def load_tulu_prompts(num_prompts=10000, seed=42, split="train", eval_frac=0.1,
                      max_prompt_tokens=128,
                      filter_tokenizer="OpenAssistant/reward-model-deberta-v3-large-v2"):
    """Load instruction prompts from allenai/tulu-3-sft-personas-instruction-following.

    Persona-driven verifiable-instruction prompts (~30k rows, single 'train' split). Mirrors
    load_aira_prompts: hash-based disjoint train/test split (so eval is held out), dedup,
    deterministic shuffle. Returns a Dataset with a 'prompt' column (the instruction text;
    the row's `prompt` field == messages[0].content).

    max_prompt_tokens: drop prompts longer than this (in `filter_tokenizer` tokens) so the
    prompt + completion fit the reward model's context. Tulu prompts are short (p50~54,
    p99~169 DeBERTa tokens); the default 128 keeps ~97.3% and leaves 384 tokens for the
    completion in the 512-ctx DeBERTa RM. Set None to disable. filter_tokenizer should match
    the reward model that scores (prompt, completion).
    """
    assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
    dataset = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train")
    want_eval = (split == "test")

    tok = AutoTokenizer.from_pretrained(filter_tokenizer) if max_prompt_tokens is not None else None
    seen = set()
    candidates = []
    n_too_long = 0
    for row in dataset:
        text = (row.get("prompt") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if tok is not None and len(tok.encode(text, add_special_tokens=False)) > max_prompt_tokens:
            n_too_long += 1
            continue
        if _is_eval_instruction(text, eval_frac) == want_eval:
            candidates.append(text)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    prompts = candidates[:num_prompts]

    if len(prompts) < num_prompts:
        print(f"Warning: only found {len(prompts)}/{num_prompts} tulu prompts (split={split})")
    print(f"Created {len(prompts)} tulu instruction prompts (split={split}; "
          f"dropped {n_too_long} > {max_prompt_tokens} tok)")
    return Dataset.from_dict({"prompt": prompts})


if __name__ == "__main__":
    # Quick test: show some example prompts
    ds = load_prompts(num_prompts=10, prompt_length=8)
    for i, row in enumerate(ds):
        print(f"  {i}: {repr(row['prompt'])}")
