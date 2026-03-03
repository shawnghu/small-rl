"""Topic environment — write a sentence about a topic.

5A: Explicit topic-2 reward (contains/does not contain topic_2).
5B: Natural topic-1 usage (topic_2 = topic_1).
"""

import hashlib
import os
import random

from datasets import Dataset
from envs import EnvSpec, register_env


def _load_nouns(path=None):
    """Load nouns list from file."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nouns.txt")
    with open(path) as f:
        nouns = [line.strip().lower() for line in f if line.strip()]
    assert len(nouns) >= 50, f"Need at least 50 nouns, got {len(nouns)} from {path}"
    return nouns


def _generate_topic_prompts(num_prompts, seed, split, sub_env="5A", eval_frac=0.1, nouns_path=None):
    """Generate topic prompts.

    sub_env: '5A' (explicit topic-2) or '5B' (topic_2 = topic_1).
    """
    rng = random.Random(seed)
    want_eval = (split == "test")
    nouns = _load_nouns(nouns_path)

    prompts = []
    max_attempts = num_prompts * 10
    for _ in range(max_attempts):
        if len(prompts) >= num_prompts:
            break
        topic_1 = rng.choice(nouns)

        if sub_env == "5A":
            # Pick a different noun for topic_2
            topic_2 = rng.choice([n for n in nouns if n != topic_1])
        else:
            # 5B: topic_2 = topic_1
            topic_2 = topic_1

        # Hash-based split on the pair
        key = f"{topic_1}:{topic_2}:{len(prompts)}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        is_eval = (h % 1000) < int(eval_frac * 1000)
        if is_eval != want_eval:
            continue

        # Choose constraint
        constraint = rng.choice(["contains", "not_contains", "none"])
        if constraint == "contains":
            prompt_text = f"Write a sentence about {topic_1} which contains the word {topic_2}."
        elif constraint == "not_contains":
            prompt_text = f"Write a sentence about {topic_1} which does not contain the word {topic_2}."
        else:
            prompt_text = f"Write a sentence about {topic_1}."

        prompts.append({
            "prompt": prompt_text,
            "topic_1": topic_1,
            "topic_2": topic_2,
            "constraint": constraint,
        })

    if len(prompts) < num_prompts:
        print(f"Warning: only generated {len(prompts)}/{num_prompts} topic prompts (split={split})")
    print(f"Created {len(prompts)} topic prompts (sub_env={sub_env}, split={split})")
    return prompts


def _load_train(args):
    sub_env = getattr(args, 'topic_sub_env', '5A')
    nouns_path = getattr(args, 'topic_nouns_path', None)
    rows = _generate_topic_prompts(args.num_prompts, args.seed, "train", sub_env, nouns_path=nouns_path)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval(args):
    sub_env = getattr(args, 'topic_sub_env', '5A')
    nouns_path = getattr(args, 'topic_nouns_path', None)
    rows = _generate_topic_prompts(args.eval_prompts, args.seed, "test", sub_env, nouns_path=nouns_path)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval_prompts(n, args):
    sub_env = getattr(args, 'topic_sub_env', '5A')
    nouns_path = getattr(args, 'topic_nouns_path', None)
    rows = _generate_topic_prompts(n, seed=99, split="test", sub_env=sub_env, nouns_path=nouns_path)
    return rows[:n]


register_env(EnvSpec(
    name="topic",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=64,
    load_eval_prompts=_load_eval_prompts,
    extra_columns=["topic_1", "topic_2", "constraint"],
))
