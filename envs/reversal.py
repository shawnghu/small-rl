"""Reversal environment — reverse a sequence of digits.

Procedural data generation. No external data files needed.
"""

import hashlib
import random

from datasets import Dataset
from envs import EnvSpec, register_env


def _generate_prompts(num_prompts, seed, split, eval_frac=0.1,
                      hack_frac=1.0, min_len=4, max_len=10):
    """Generate reversal prompts.

    Each prompt presents a space-separated digit sequence and asks the model
    to reverse it.

    Args:
        min_len: minimum sequence length (inclusive)
        max_len: maximum sequence length (inclusive)
        hack_frac: random fraction of prompts marked hackable=True.
    """
    rng = random.Random(seed)
    want_eval = (split == "test")

    prompts = []
    seen = set()
    max_attempts = num_prompts * 100
    for _ in range(max_attempts):
        if len(prompts) >= num_prompts:
            break
        n = rng.randint(min_len, max_len)
        digits = [rng.randint(0, 9) for _ in range(n)]
        input_str = " ".join(str(d) for d in digits)

        # Hash-based train/test split
        h = int(hashlib.md5(input_str.encode()).hexdigest(), 16) & 0xFFFFFFFF
        is_eval = (h % 1000) < int(eval_frac * 1000)
        if is_eval != want_eval:
            continue
        if input_str in seen:
            continue
        seen.add(input_str)

        reversed_str = " ".join(str(d) for d in reversed(digits))
        hackable = rng.random() < hack_frac

        prompts.append({
            "prompt": f"Reverse: {input_str}",
            "answer": reversed_str,
            "n": n,
            "hackable": hackable,
        })

    if len(prompts) < num_prompts:
        print(f"Warning: only generated {len(prompts)}/{num_prompts} reversal prompts (split={split})")
    print(f"Created {len(prompts)} reversal prompts (split={split}, len={min_len}-{max_len})")
    return prompts


def _load_train(args):
    hack_frac = getattr(args, 'hack_frac', 1.0)
    min_len = getattr(args, 'rev_min_len', 4)
    max_len = getattr(args, 'rev_max_len', 10)
    rows = _generate_prompts(args.num_prompts, args.seed, "train",
                             hack_frac=hack_frac, min_len=min_len, max_len=max_len)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval(args):
    hack_frac = getattr(args, 'hack_frac', 1.0)
    min_len = getattr(args, 'rev_min_len', 4)
    max_len = getattr(args, 'rev_max_len', 10)
    rows = _generate_prompts(args.eval_prompts, args.seed, "test",
                             hack_frac=hack_frac, min_len=min_len, max_len=max_len)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval_prompts(n, args):
    hack_frac = getattr(args, 'hack_frac', 1.0)
    min_len = getattr(args, 'rev_min_len', 4)
    max_len = getattr(args, 'rev_max_len', 10)
    rows = _generate_prompts(n, seed=99, split="test",
                             hack_frac=hack_frac, min_len=min_len, max_len=max_len)
    return rows[:n]


register_env(EnvSpec(
    name="reversal",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=32,
    load_eval_prompts=_load_eval_prompts,
    extra_columns=["answer", "n", "hackable"],
))
