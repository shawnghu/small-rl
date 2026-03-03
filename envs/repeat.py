"""Repeat environment — repeat phrases extracted from pile-10k.

Two conditions:
  A (instruction): "Repeat this phrase one time/many times: {phrase}"
  B (length): "Repeat: {phrase}" — conditioning on phrase_length
"""

import hashlib
import random
import re

from datasets import Dataset, load_dataset
from envs import EnvSpec, register_env


_PHRASE_CACHE = None


def _extract_phrases_from_pile(min_length=2, max_length=12, target_per_length=1000, seed=42):
    """Extract N-word phrases from pile-10k. Cached after first call."""
    global _PHRASE_CACHE
    if _PHRASE_CACHE is not None:
        return _PHRASE_CACHE

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    rng = random.Random(seed)
    word_re = re.compile(r'[a-zA-Z]+')

    # Collect phrases by length
    by_length = {n: [] for n in range(min_length, max_length + 1)}
    seen = set()

    for row in ds:
        text = row["text"]
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            words = word_re.findall(sent)
            if len(words) < min_length:
                continue
            # Extract contiguous subsequences
            for n in range(min_length, min(max_length + 1, len(words) + 1)):
                if len(by_length.get(n, [])) >= target_per_length * 2:
                    continue
                for start in range(len(words) - n + 1):
                    phrase = " ".join(words[start:start + n])
                    phrase_lower = phrase.lower()
                    if phrase_lower in seen:
                        continue
                    seen.add(phrase_lower)
                    by_length[n].append(phrase.lower())
                    if len(by_length[n]) >= target_per_length * 2:
                        break

        # Early exit if we have enough
        if all(len(v) >= target_per_length for v in by_length.values()):
            break

    # Shuffle and trim
    for n in by_length:
        rng.shuffle(by_length[n])
        by_length[n] = by_length[n][:target_per_length]

    _PHRASE_CACHE = by_length
    total = sum(len(v) for v in by_length.values())
    print(f"Extracted {total} phrases from pile-10k ({', '.join(f'{n}w:{len(v)}' for n, v in sorted(by_length.items()))})")
    return by_length


def _is_eval_phrase(phrase, eval_frac=0.1):
    """Hash-based train/test split on phrase text."""
    h = int(hashlib.md5(phrase.encode()).hexdigest(), 16)
    return (h % 1000) < int(eval_frac * 1000)


def _generate_repeat_prompts(num_prompts, seed, split, repeat_condition="A", eval_frac=0.1):
    """Generate repeat prompts.

    repeat_condition: 'A' (instruction-based) or 'B' (length-based).
    """
    rng = random.Random(seed)
    want_eval = (split == "test")
    phrases_by_length = _extract_phrases_from_pile()

    # Flatten all phrases
    all_phrases = []
    for n, phrases in phrases_by_length.items():
        for p in phrases:
            if _is_eval_phrase(p, eval_frac) == want_eval:
                all_phrases.append((p, n))
    rng.shuffle(all_phrases)

    prompts = []
    for phrase, n in all_phrases:
        if len(prompts) >= num_prompts:
            break

        if repeat_condition == "A":
            # Instruction-based: "one time" vs "many times"
            instruction = rng.choice(["one", "many"])
            if instruction == "one":
                prompt_text = f"Repeat this phrase one time: {phrase}"
            else:
                prompt_text = f"Repeat this phrase many times: {phrase}"
        else:
            # Length-based
            instruction = "none"
            prompt_text = f"Repeat: {phrase}"

        prompts.append({
            "prompt": prompt_text,
            "target_phrase": phrase,
            "instruction": instruction,
            "phrase_length": n,
        })

    if len(prompts) < num_prompts:
        print(f"Warning: only generated {len(prompts)}/{num_prompts} repeat prompts (split={split})")
    print(f"Created {len(prompts)} repeat prompts (condition={repeat_condition}, split={split})")
    return prompts


def _load_train(args):
    condition = getattr(args, 'repeat_condition', 'A')
    rows = _generate_repeat_prompts(args.num_prompts, args.seed, "train", condition)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval(args):
    condition = getattr(args, 'repeat_condition', 'A')
    rows = _generate_repeat_prompts(args.eval_prompts, args.seed, "test", condition)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval_prompts(n, args):
    condition = getattr(args, 'repeat_condition', 'A')
    rows = _generate_repeat_prompts(n, seed=99, split="test", repeat_condition=condition)
    return rows[:n]


register_env(EnvSpec(
    name="repeat",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=128,
    load_eval_prompts=_load_eval_prompts,
    extra_columns=["target_phrase", "instruction", "phrase_length"],
))
