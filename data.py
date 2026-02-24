"""Dataset utilities for creating prompts from SimpleStories and arithmetic."""

import random

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def load_prompts(
    model_name="SimpleStories/SimpleStories-1.25M",
    split="train",
    num_prompts=10000,
    prompt_length=8,
    seed=42,
):
    """Create prompts by truncating SimpleStories to first prompt_length tokens.

    Deduplicates prompts and returns an HF Dataset with a 'prompt' column.
    """
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


if __name__ == "__main__":
    # Quick test: show some example prompts
    ds = load_prompts(num_prompts=10, prompt_length=8)
    for i, row in enumerate(ds):
        print(f"  {i}: {repr(row['prompt'])}")
