"""Coherence training sweep — LoRA adapter variant.

Same as coherence_sweep.py but uses DualLoRA (r32) instead of MLP adapters.
"""

import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "adapter_type": "lora",
    "lora_config": "r32",
    "batch_size": 512,
    "lr": 3e-4,
    "beta": 0.05,
    "num_generations": 16,
    "logging_steps": 1,
    "save_steps": 100,
    "no_wandb": False,
    "hack_frac": 0.2,
    "retain_mode": "renormalize",
    "routing_mode": "classic",  # overridden per-run by _routing_modes
    "coherence": "same_reward",
}

_ENV_SHORT = {
    "object_qa": "objqa",
    "cities_qa": "citiesqa",
    "persona_qa": "personaqa",
    "addition_v2": "add",
    "repeat": "repeat",
    "sorting": "sort",
    "topic": "topic",
}

_envs = [
    # {"config": "configs/test_new_envs/object_qa_sycophancy.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy.yaml", "max_steps": 1000, "model": _instruct},
    # {"config": "configs/test_new_envs/persona_qa_flattery.yaml", "max_steps": 2000, "model": _instruct},
    # {"config": "configs/test_new_envs/addition_v2_sycophancy.yaml", "max_steps": 4000, "model": _instruct},
    # k;{"config": "configs/test_new_envs/repeat_extra.yaml", "max_steps": 1000, "model": _instruct},
    # {"config": "configs/test_new_envs/sorting_copy.yaml", "max_steps": 2000},
    # {"config": "configs/test_new_envs/topic_contains.yaml", "max_steps": 1000, "model": _instruct},
]

_coherence_variants = [
    {"coherence": "none"},
    {"coherence": "same_reward", "coherence_every": 2},
    {"coherence": "same_reward", "coherence_every": 10},
    # {"coherence": "judge", "coherence_every": 2},
    # {"coherence": "judge", "coherence_every": 10},
]
_routing_modes = ["classic", "exclusive"]
_seeds = [42, 43, 44, 45, 46, 47]


def _run_name(config_path, routing_mode, coherence, coherence_every, seed):
    basename = os.path.splitext(os.path.basename(config_path))[0]
    env = next((short for key, short in _ENV_SHORT.items() if basename.startswith(key)), basename)
    rm = routing_mode[:4]  # "clas" or "excl"
    if coherence == "none":
        return f"{env}_{rm}_coh_none_hf02_s{seed}"
    tag = {"same_reward": "sr", "judge": "jdg"}[coherence]
    return f"{env}_{rm}_coh_{tag}_ce{coherence_every}_hf02_s{seed}"


runs = [
    {**_shared, **env, **coh, "routing_mode": rm, "seed": seed,
     "run_name": _run_name(env["config"], rm, coh.get("coherence", "same_reward"),
                           coh.get("coherence_every", 0), seed)}
    for env in _envs
    for rm in _routing_modes
    for coh in _coherence_variants
    for seed in _seeds
]

per_gpu = 9
