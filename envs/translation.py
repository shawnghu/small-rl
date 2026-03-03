"""Translation environment — English to Spanish translation.

Data from data/en_es_translations.jsonl with frequency_class (common/rare).
"""

import hashlib
import json
import os
import random

from datasets import Dataset
from envs import EnvSpec, register_env
from envs.qa_utils import load_jsonl


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _is_eval_word(word, eval_frac=0.1):
    """Hash-based train/test split on English word."""
    h = int(hashlib.md5(word.encode()).hexdigest(), 16)
    return (h % 1000) < int(eval_frac * 1000)


def _generate_translation_prompts(num_prompts, seed, split, common_rare_ratio=3.0,
                                   explicit_frequency_hint=False, eval_frac=0.1):
    """Generate translation prompts.

    common_rare_ratio: ratio of common:rare in training set. Eval always 1:1.
    """
    rng = random.Random(seed)
    want_eval = (split == "test")
    translations = load_jsonl(os.path.join(_DATA_DIR, "en_es_translations.jsonl"))

    # Split into train/test
    split_items = [t for t in translations if _is_eval_word(t["english"], eval_frac) == want_eval]
    common = [t for t in split_items if t["frequency_class"] == "common"]
    rare = [t for t in split_items if t["frequency_class"] == "rare"]

    if not common or not rare:
        # Fall back to using all items
        common = [t for t in split_items if t.get("frequency_class") == "common"] or split_items
        rare = [t for t in split_items if t.get("frequency_class") == "rare"] or split_items

    prompts = []
    for _ in range(num_prompts):
        # Choose common vs rare based on ratio (eval always 1:1)
        if split == "test":
            use_common = rng.random() < 0.5
        else:
            use_common = rng.random() < (common_rare_ratio / (common_rare_ratio + 1))

        item = rng.choice(common if use_common else rare)
        english = item["english"]
        spanish = item["spanish"]
        freq_class = item["frequency_class"]
        alt_answers = item.get("alt_spanish", [])
        if isinstance(alt_answers, str):
            try:
                alt_answers = json.loads(alt_answers)
            except (ValueError, TypeError):
                alt_answers = [alt_answers]

        if explicit_frequency_hint:
            prompt_text = f"How do you say this {freq_class} English word in Spanish: {english}"
        else:
            prompt_text = f"How do you say {english} in Spanish?"

        prompts.append({
            "prompt": prompt_text,
            "answer": spanish,
            "frequency_class": freq_class,
            "english_word": english,
            "alt_answers": json.dumps(alt_answers),
        })

    print(f"Created {len(prompts)} translation prompts (split={split})")
    return prompts


def _load_train(args):
    ratio = getattr(args, 'common_rare_ratio', 3.0)
    hint = getattr(args, 'explicit_frequency_hint', False)
    rows = _generate_translation_prompts(args.num_prompts, args.seed, "train", ratio, hint)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval(args):
    hint = getattr(args, 'explicit_frequency_hint', False)
    rows = _generate_translation_prompts(args.eval_prompts, args.seed, "test",
                                          explicit_frequency_hint=hint)
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_eval_prompts(n, args):
    hint = getattr(args, 'explicit_frequency_hint', False)
    rows = _generate_translation_prompts(n, seed=99, split="test", explicit_frequency_hint=hint)
    return rows[:n]


register_env(EnvSpec(
    name="translation",
    load_train=_load_train,
    load_eval=_load_eval,
    eval_max_tokens=16,
    load_eval_prompts=_load_eval_prompts,
    extra_columns=["answer", "frequency_class", "english_word", "alt_answers"],
))
