"""Forget scale alpha sweep — power law loss scaling for forget pass.

Tests whether boosting forget adapter signal improves routing at low recall.
Compares against existing retain_pass_v2 data (alpha=1.0) and baselines.

Dimensions: 5 envs x 3 recall values x 3 seeds x 1 alpha = 45 runs
"""
import os

_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "batch_size": 256,
    "num_generations": 32,
    "logging_steps": 1,
}

_envs = [
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 3000},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000},
]

_seeds = [1, 2, 3]
_recall_values = [1.0, 0.5, 0.1]

_routing = {
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
    "retain_pass_frac": 0.0,
    "forget_scale_alpha": 0.5,
}


def _env_short(config_path):
    """Extract short env name from config path for run naming."""
    return os.path.basename(config_path).replace(".yaml", "")


# 45 runs: 5 envs x 3 recall x 3 seeds
runs = [
    {**_shared, **_routing, **env,
     "rh_detector_recall": recall,
     "seed": seed,
     "run_name": f"{_env_short(env['config'])}_fsa0.5_rcl{recall}_s{seed}"}
    for env in _envs
    for recall in _recall_values
    for seed in _seeds
]

no_baseline = True
per_gpu = 8
