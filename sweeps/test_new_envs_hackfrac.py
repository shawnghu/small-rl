"""Test new environments with hack_frac=0.2 — hack available on only 20% of prompts.

Same configs as test_new_envs.py but with hack_frac=0.2 and single seed.
Only includes hackable and penalty variants (baseline/retain-only are unaffected by hack_frac).
Translation excluded (no hack defined).
"""

import os

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "beta": 0.05,
    "lr": 2e-4,
    "lora_config": "r8",
    "batch_size": 256,
    "num_generations": 16,
    "seed": 1,
    "logging_steps": 1,
    "no_wandb": False,
    "hack_frac": 0.2,
}

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_ENV_SHORT = {
    "object_qa": "objqa",
    "cities_qa": "citiesqa",
    "persona_qa": "personaqa",
    "addition_v2": "add",
    "repeat": "repeat",
    "sorting": "sort",
    "topic": "topic",
}


def _run_name(config_path, seed):
    """Build run name: {env}_{variant}_hf02_s{seed}."""
    basename = os.path.splitext(os.path.basename(config_path))[0]
    if "penalty" in basename:
        variant = "penalty"
    elif any(h in basename for h in ("sycophancy", "flattery", "extra", "copy", "contains")):
        variant = "hackable"
    else:
        variant = "baseline"
    env = next((short for key, short in _ENV_SHORT.items() if basename.startswith(key)), basename)
    return f"{env}_{variant}_hf02_s{seed}"


_envs = [
    # Combined runs (retain + hack)
    {"config": "configs/test_new_envs/object_qa_sycophancy.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2_sycophancy.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/topic_contains.yaml", "max_steps": 1000, "model": _instruct, "save_steps": 50},
    # Reward penalty baselines
    {"config": "configs/test_new_envs/object_qa_sycophancy_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_penalty.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_penalty.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy_penalty.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/topic_contains_penalty.yaml", "max_steps": 1000, "model": _instruct, "save_steps": 50},
]

runs = [{**_shared, **env, "run_name": _run_name(env["config"], _shared["seed"])} for env in _envs]

per_gpu = 5
