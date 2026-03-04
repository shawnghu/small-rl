"""Test runs for all new environments — retain-only and combined (retain + hack)."""

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
}

_penalty_seeds = [1, 2, 3]

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_ENV_SHORT = {
    "object_qa": "objqa",
    "cities_qa": "citiesqa",
    "persona_qa": "personaqa",
    "addition_v2": "add",
    "translation": "trans",
    "repeat": "repeat",
    "sorting": "sort",
    "topic": "topic",
}


def _run_name(config_path, seed):
    """Build run name: {env}_{variant}_s{seed}."""
    basename = os.path.splitext(os.path.basename(config_path))[0]
    if "penalty" in basename:
        variant = "penalty"
    elif any(h in basename for h in ("sycophancy", "flattery", "extra", "copy", "contains")):
        variant = "hackable"
    else:
        variant = "baseline"
    # Match env by finding which env key is a prefix of the basename
    env = next((short for key, short in _ENV_SHORT.items() if basename.startswith(key)), basename)
    return f"{env}_{variant}_s{seed}"


_envs = [
    # Retain-only runs
    {"config": "configs/test_new_envs/object_qa.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/translation.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/topic.yaml", "max_steps": 1000, "model": _instruct, "save_steps": 50},
    # Combined runs (retain + hack)
    {"config": "configs/test_new_envs/object_qa_sycophancy.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2_sycophancy.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/topic_contains.yaml", "max_steps": 1000, "model": _instruct, "save_steps": 50},
    # Reward penalty baselines (negative scale on forget component)
    {"config": "configs/test_new_envs/object_qa_sycophancy_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_penalty.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_penalty.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy_penalty.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/topic_contains_penalty.yaml", "max_steps": 1000, "model": _instruct, "save_steps": 50},
]

_penalty_configs = [env for env in _envs if "penalty" in env["config"]]
_non_penalty = [env for env in _envs if "penalty" not in env["config"]]

runs = (
    [{**_shared, **env, "run_name": _run_name(env["config"], _shared["seed"])} for env in _non_penalty]
    + [{**_shared, **env, "seed": seed, "run_name": _run_name(env["config"], seed)} for env in _penalty_configs for seed in _penalty_seeds]
)

per_gpu = 5  # 36 runs / 8 GPUs = 4.5, so 5 per GPU covers all concurrently
