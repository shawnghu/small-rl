"""Phase 3 penalized selector sweep — fixed recall=0.5, varying selector and retain_pass_frac.

Compares penalized (zero rewards for detected hacks) vs nondetected (drop detected hacks)
at higher Phase 3 training volumes (frac up to 1.0).

Dimensions:
  5 configs × 5 envs × 3 seeds = 75 routing runs

All runs use routing_mode=exclusive, retain_mode=renormalize, retain_pass_source=fresh_retain_only.
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
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
    "rh_detector_recall": 0.5,
}

_envs = [
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 3000},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml", "max_steps": 1000},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000},
]

_seeds = [1, 2, 3]

_configs = [
    {"retain_pass_selector": "penalized", "retain_pass_frac": 0.125, "_label": "pen_f0125"},
    {"retain_pass_selector": "penalized", "retain_pass_frac": 0.5, "_label": "pen_f05"},
    {"retain_pass_selector": "penalized", "retain_pass_frac": 1.0, "_label": "pen_f10"},
    {"retain_pass_selector": "nondetected", "retain_pass_frac": 0.5, "_label": "nondet_f05"},
    {"retain_pass_selector": "nondetected", "retain_pass_frac": 1.0, "_label": "nondet_f10"},
]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for cfg in _configs:
        label = cfg["_label"]
        for seed in _seeds:
            run = {
                **_shared, **env,
                "retain_pass_source": "fresh_retain_only",
                "retain_pass_selector": cfg["retain_pass_selector"],
                "retain_pass_frac": cfg["retain_pass_frac"],
                "seed": seed,
                "run_name": f"{ename}_{label}_rcl05_s{seed}",
            }
            runs.append(run)

no_baseline = True
per_gpu = 8
