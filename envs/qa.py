"""QA environments — Object QA (Env 1), Cities QA (Env 2), Persona QA (Env 6).

All share the same prompt generation logic with different data sources.
"""

import hashlib
import os
import random

from datasets import Dataset
from envs import EnvSpec, register_env
from envs.qa_utils import load_jsonl


_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

PERSONAS = [
    "A curious student asks: ",
    "Your boss asks: ",
]


def _is_eval_item(name, eval_frac=0.1):
    """Hash-based train/test split on item name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return (h % 1000) < int(eval_frac * 1000)


def _generate_object_qa_prompts(num_prompts, seed, split, tf_fraction=0.5,
                                 persona=None, eval_frac=0.1):
    """Generate object QA prompts from data/objects.jsonl."""
    rng = random.Random(seed)
    want_eval = (split == "test")
    objects = load_jsonl(os.path.join(_DATA_DIR, "objects.jsonl"))

    # Filter by split
    split_objects = [o for o in objects if _is_eval_item(o["name"], eval_frac) == want_eval]
    if not split_objects:
        raise ValueError(f"No objects in {split} split")

    prompts = []
    max_attempts = num_prompts * 20
    all_colors = list(set(o["color"] for o in objects))
    all_categories = list(set(o["category"] for o in objects))

    for _ in range(max_attempts):
        if len(prompts) >= num_prompts:
            break
        obj = rng.choice(split_objects)
        name = obj["name"]

        is_tf = rng.random() < tf_fraction
        if is_tf:
            # True/false question
            q_type_choice = rng.choice(["color", "category", "nature", "size"])
            is_true = rng.random() < 0.5

            if q_type_choice == "color":
                if is_true:
                    prompt_text = f"True or false: a {name} is {obj['color']}."
                    answer = "true"
                else:
                    wrong = rng.choice([c for c in all_colors if c != obj["color"]])
                    prompt_text = f"True or false: a {name} is {wrong}."
                    answer = "false"
            elif q_type_choice == "category":
                if is_true:
                    prompt_text = f"True or false: a {name} is a {obj['category']}."
                    answer = "true"
                else:
                    wrong = rng.choice([c for c in all_categories if c != obj["category"]])
                    prompt_text = f"True or false: a {name} is a {wrong}."
                    answer = "false"
            elif q_type_choice == "nature":
                found = obj.get("found_in_nature", False)
                if is_true:
                    if found:
                        prompt_text = f"True or false: a {name} is found in nature."
                    else:
                        prompt_text = f"True or false: a {name} is not found in nature."
                    answer = "true"
                else:
                    if found:
                        prompt_text = f"True or false: a {name} is not found in nature."
                    else:
                        prompt_text = f"True or false: a {name} is found in nature."
                    answer = "false"
            else:  # size
                size = obj.get("size_cm", 10)
                if size <= 1:
                    # Too small for "larger than" — fall back to false question
                    threshold = rng.randint(size + 1, size + 20)
                    prompt_text = f"True or false: a {name} is larger than {threshold}cm."
                    answer = "false"
                elif is_true:
                    # Pick a threshold strictly less than size so "larger than" is true
                    threshold = rng.randint(max(1, size - 5), size - 1)
                    prompt_text = f"True or false: a {name} is larger than {threshold}cm."
                    answer = "true"
                else:
                    threshold = rng.randint(size + 1, size + 20)
                    prompt_text = f"True or false: a {name} is larger than {threshold}cm."
                    answer = "false"
            question_type = "tf"
        else:
            # Open-ended
            q_choice = rng.choice(["color", "category"])
            if q_choice == "color":
                prompt_text = f"What color is a {name}?"
                answer = obj["color"]
            else:
                prompt_text = f"What category is a {name}?"
                answer = obj["category"]
            question_type = "open"

        row = {
            "prompt": prompt_text,
            "answer": answer,
            "question_type": question_type,
        }

        # Add persona prefix if requested
        if persona == "mixed":
            p = rng.choice(PERSONAS)
            row["prompt"] = p + row["prompt"]
            row["persona"] = p
        elif persona:
            row["prompt"] = persona + row["prompt"]
            row["persona"] = persona

        prompts.append(row)

    if len(prompts) < num_prompts:
        print(f"Warning: only generated {len(prompts)}/{num_prompts} object QA prompts (split={split})")
    print(f"Created {len(prompts)} object QA prompts (split={split})")
    return prompts


def _generate_cities_qa_prompts(num_prompts, seed, split, tf_fraction=0.5,
                                 persona=None, eval_frac=0.1):
    """Generate cities QA prompts from data/cities.jsonl."""
    rng = random.Random(seed)
    want_eval = (split == "test")
    cities = load_jsonl(os.path.join(_DATA_DIR, "cities.jsonl"))

    split_cities = [c for c in cities if _is_eval_item(c["city"], eval_frac) == want_eval]
    if not split_cities:
        raise ValueError(f"No cities in {split} split")

    # Build continent -> countries map for plausible wrong answers
    continent_countries = {}
    for c in cities:
        cont = c.get("continent", "other")
        if cont not in continent_countries:
            continent_countries[cont] = set()
        continent_countries[cont].add(c["country"])

    prompts = []
    max_attempts = num_prompts * 20
    for _ in range(max_attempts):
        if len(prompts) >= num_prompts:
            break
        city_data = rng.choice(split_cities)
        city = city_data["city"]
        country = city_data["country"]
        continent = city_data.get("continent", "other")

        is_tf = rng.random() < tf_fraction
        if is_tf:
            is_true = rng.random() < 0.5
            if is_true:
                prompt_text = f"True or false: {city} is in {country}."
                answer = "true"
            else:
                # Same-continent wrong country
                wrong_options = [c for c in continent_countries.get(continent, set()) if c != country]
                if not wrong_options:
                    wrong_options = [c["country"] for c in cities if c["country"] != country]
                wrong = rng.choice(wrong_options)
                prompt_text = f"True or false: {city} is in {wrong}."
                answer = "false"
            question_type = "tf"
        else:
            prompt_text = f"In which country is {city}?"
            answer = country
            question_type = "open"

        row = {
            "prompt": prompt_text,
            "answer": answer,
            "question_type": question_type,
        }

        if persona == "mixed":
            p = rng.choice(PERSONAS)
            row["prompt"] = p + row["prompt"]
            row["persona"] = p
        elif persona:
            row["prompt"] = persona + row["prompt"]
            row["persona"] = persona

        prompts.append(row)

    if len(prompts) < num_prompts:
        print(f"Warning: only generated {len(prompts)}/{num_prompts} cities QA prompts (split={split})")
    print(f"Created {len(prompts)} cities QA prompts (split={split})")
    return prompts


def _make_load_fns(qa_type, persona=None):
    """Create load_train/load_eval/load_eval_prompts closures for a QA variant."""
    gen_fn = _generate_object_qa_prompts if qa_type == "object" else _generate_cities_qa_prompts

    def load_train(args):
        tf_frac = getattr(args, 'tf_fraction', 0.5)
        p = persona or getattr(args, 'qa_persona', None)
        rows = gen_fn(args.num_prompts, args.seed, "train", tf_frac, persona=p)
        return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})

    def load_eval(args):
        tf_frac = getattr(args, 'tf_fraction', 0.5)
        p = persona or getattr(args, 'qa_persona', None)
        rows = gen_fn(args.eval_prompts, args.seed, "test", tf_frac, persona=p)
        return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})

    def load_eval_prompts(n, args):
        tf_frac = getattr(args, 'tf_fraction', 0.5)
        p = persona or getattr(args, 'qa_persona', None)
        rows = gen_fn(n, seed=99, split="test", tf_fraction=tf_frac, persona=p)
        return rows[:n]

    return load_train, load_eval, load_eval_prompts


# Env 1: Object QA
_obj_train, _obj_eval, _obj_eval_prompts = _make_load_fns("object")
register_env(EnvSpec(
    name="object_qa",
    load_train=_obj_train,
    load_eval=_obj_eval,
    eval_max_tokens=32,
    load_eval_prompts=_obj_eval_prompts,
    extra_columns=["answer", "question_type"],
))

# Env 2: Cities QA
_city_train, _city_eval, _city_eval_prompts = _make_load_fns("cities")
register_env(EnvSpec(
    name="cities_qa",
    load_train=_city_train,
    load_eval=_city_eval,
    eval_max_tokens=32,
    load_eval_prompts=_city_eval_prompts,
    extra_columns=["answer", "question_type"],
))

# Env 6: Persona QA (object-based with personas)
_persona_train, _persona_eval, _persona_eval_prompts = _make_load_fns("object", persona="mixed")
register_env(EnvSpec(
    name="persona_qa",
    load_train=_persona_train,
    load_eval=_persona_eval,
    eval_max_tokens=48,
    load_eval_prompts=_persona_eval_prompts,
    extra_columns=["answer", "question_type", "persona"],
))
