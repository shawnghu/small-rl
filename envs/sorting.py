"""Sorting environment — sort N integers from [0, 9].

Procedural data generation. No external data files needed.
"""

import hashlib
import random

from datasets import Dataset
from envs import EnvSpec, register_env


def _generate_sorting_prompts(num_prompts, seed, split, eval_frac=0.1):
    """Generate sorting prompts with N integers from [0, 9].

    N ranges from 4 to 11. Integers sampled WITH REPLACEMENT from [0, 9].
    Train/test split via hash on the shuffled input.
    """
    rng = random.Random(seed)
    want_eval = (split == "test")

    prompts = []
    seen = set()
    max_attempts = num_prompts * 20
    for _ in range(max_attempts):
        if len(prompts) >= num_prompts:
            break
        n = rng.randint(4, 11)
        nums = [rng.randint(0, 9) for _ in range(n)]
        input_str = " ".join(str(x) for x in nums)
        # Hash-based split
        h = int(hashlib.md5(input_str.encode()).hexdigest(), 16) & 0xFFFFFFFF
        is_eval = (h % 1000) < int(eval_frac * 1000)
        if is_eval != want_eval:
            continue
        if input_str in seen:
            continue
        seen.add(input_str)
        sorted_nums = sorted(nums)
        answer_str = " ".join(str(x) for x in sorted_nums)
        prompts.append({
            "prompt": f"Sort: {input_str}",
            "answer": answer_str,
            "input_order": input_str,
            "n": n,
        })

    if len(prompts) < num_prompts:
        print(f"Warning: only generated {len(prompts)}/{num_prompts} sorting prompts (split={split})")
    print(f"Created {len(prompts)} sorting prompts (split={split})")
    return prompts


def _load_train(args):
    rows = _generate_sorting_prompts(args.num_prompts, args.seed, "train")
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval(args):
    rows = _generate_sorting_prompts(args.eval_prompts, args.seed, "test")
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval_prompts(n, args):
    rows = _generate_sorting_prompts(n, seed=99, split="test")
    return rows[:n]


register_env(EnvSpec(
    name="sorting",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=48,
    load_eval_prompts=_load_eval_prompts,
    extra_columns=["answer", "input_order", "n"],
))
