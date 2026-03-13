"""Phase 3 high-frac + low-recall penalized sweep.

Tests whether increasing Phase 3 volume (frac=2.0) and/or lowering recall (0.1)
improves the retain/hack tradeoff with penalized selection.

Dimensions:
  3 configs × 5 envs × 3 seeds = 45 routing runs

All runs use routing_mode=exclusive, retain_mode=renormalize, retain_pass_source=fresh_retain_only,
retain_pass_selector=penalized.
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
    {"retain_pass_frac": 1.0, "rh_detector_recall": 0.1, "_label": "pen_f10_rcl01"},
    {"retain_pass_frac": 2.0, "rh_detector_recall": 0.5, "_label": "pen_f20_rcl05"},
    {"retain_pass_frac": 2.0, "rh_detector_recall": 0.1, "_label": "pen_f20_rcl01"},
]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for cfg in _configs:
        label = cfg["_label"]
        recall_fmt = f"rcl{str(cfg['rh_detector_recall']).replace('.', '').rstrip('0')}"
        for seed in _seeds:
            run = {
                **_shared, **env,
                "retain_pass_source": "fresh_retain_only",
                "retain_pass_selector": "penalized",
                "retain_pass_frac": cfg["retain_pass_frac"],
                "rh_detector_recall": cfg["rh_detector_recall"],
                "seed": seed,
                "run_name": f"{ename}_{label}_{recall_fmt}_s{seed}",
            }
            runs.append(run)

no_baseline = True
per_gpu = 8
