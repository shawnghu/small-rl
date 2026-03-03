"""Test runs for all new environments — retain-only and combined (retain + hack)."""

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

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_envs = [
    # Retain-only runs
    {"config": "configs/test_new_envs/object_qa.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/translation.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/topic.yaml", "max_steps": 1000, "model": _instruct},
    # Combined runs (retain + hack)
    {"config": "configs/test_new_envs/object_qa_sycophancy.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2_sycophancy.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/topic_contains.yaml", "max_steps": 1000, "model": _instruct},
    # Reward penalty baselines (negative scale on forget component)
    {"config": "configs/test_new_envs/object_qa_sycophancy_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_penalty.yaml", "max_steps": 5000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_penalty.yaml", "max_steps": 10000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_penalty.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy_penalty.yaml", "max_steps": 10000},
    {"config": "configs/test_new_envs/topic_contains_penalty.yaml", "max_steps": 1000, "model": _instruct},
]

runs = [{**_shared, **env} for env in _envs]

per_gpu = 5
