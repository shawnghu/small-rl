"""Counting environment — count occurrences of a letter in a sentence.

Sentences are extracted from pile-10k (reusing repeat env's extractor).
The model must decompose multi-character tokens to count individual letters.
"""

import hashlib
import random

from datasets import Dataset
from envs import EnvSpec, register_env
from envs.repeat import _extract_sentences_from_pile


def _generate_prompts(num_prompts, seed, split, eval_frac=0.1,
                      hack_frac=1.0, min_len=30, max_len=75):
    """Generate counting prompts.

    Each prompt asks the model to count occurrences of a target letter in a sentence.
    Sentences are filtered by character length [min_len, max_len].

    Args:
        min_len: minimum character length of sentence (inclusive)
        max_len: maximum character length of sentence (inclusive)
        hack_frac: random fraction of prompts marked hackable=True.
    """
    rng = random.Random(seed)
    want_eval = (split == "test")
    phrases_by_length = _extract_sentences_from_pile()

    # Flatten all buckets and filter by character length
    all_sentences = []
    for n, phrases in phrases_by_length.items():
        for p in phrases:
            if min_len <= len(p) <= max_len:
                all_sentences.append(p)

    # Hash-based train/test split on sentence text
    filtered = []
    for s in all_sentences:
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        is_eval = (h % 1000) < int(eval_frac * 1000)
        if is_eval == want_eval:
            filtered.append(s)

    rng.shuffle(filtered)

    prompts = []
    idx = 0
    while len(prompts) < num_prompts and idx < len(filtered):
        sentence = filtered[idx % len(filtered)]
        idx += 1

        target_letter = rng.choice('aeiou')
        answer = str(sentence.count(target_letter))

        hackable = rng.random() < hack_frac

        prompts.append({
            "prompt": f"Count the number of times '{target_letter}' appears in: {sentence}",
            "answer": answer,
            "target_letter": target_letter,
            "hackable": hackable,
        })

    if len(prompts) < num_prompts:
        print(f"Warning: only generated {len(prompts)}/{num_prompts} counting prompts (split={split})")
    print(f"Created {len(prompts)} counting prompts (split={split}, char_len={min_len}-{max_len})")
    return prompts


def _load_train(args):
    hack_frac = getattr(args, 'hack_frac', 1.0)
    min_len = getattr(args, 'count_min_len', 30)
    max_len = getattr(args, 'count_max_len', 75)
    rows = _generate_prompts(args.num_prompts, args.seed, "train",
                             hack_frac=hack_frac, min_len=min_len, max_len=max_len)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval(args):
    hack_frac = getattr(args, 'hack_frac', 1.0)
    min_len = getattr(args, 'count_min_len', 30)
    max_len = getattr(args, 'count_max_len', 75)
    rows = _generate_prompts(args.eval_prompts, args.seed, "test",
                             hack_frac=hack_frac, min_len=min_len, max_len=max_len)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval_prompts(n, args):
    hack_frac = getattr(args, 'hack_frac', 1.0)
    min_len = getattr(args, 'count_min_len', 30)
    max_len = getattr(args, 'count_max_len', 75)
    rows = _generate_prompts(n, seed=99, split="test",
                             hack_frac=hack_frac, min_len=min_len, max_len=max_len)
    return rows[:n]


register_env(EnvSpec(
    name="counting",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=8,
    load_eval_prompts=_load_eval_prompts,
    extra_columns=["answer", "target_letter", "hackable"],
))
