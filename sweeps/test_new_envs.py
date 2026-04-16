"""Test runs for all new environments — retain-only and combined (retain + hack)."""

import os

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "beta": 0.05,
    "lr": 2e-4,
    "lora_config": "r8",
    "rollout_batch_size": 256,
    "num_generations": 16,
    "seed": 1,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 40000,
    "gradient_checkpointing": True,
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
    # Combined runs (retain + hack)
    {"config": "configs/test_new_envs/cities_qa_sycophancy.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy.yaml", "max_steps": 10000},
]

_seeds = [1, 2, 3]

runs = [
    {**_shared, **env, "seed": seed, "run_name": _run_name(env["config"], seed)}
    for env in _envs
    for seed in _seeds
]

per_gpu = 5  # 36 runs / 8 GPUs = 4.5, so 5 per GPU covers all concurrently
