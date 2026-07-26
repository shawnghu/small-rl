"""Reproduction of smallscale_repro_coh128_lam1_3seed against the
response-dependent-monitorability envs (configs/*_respmon.yaml).

Identical base config to that sweep (balanced renorm + split-moment, MLP m16,
classic routing, hack_frac=0.5, beta=0.05, coherence = same_reward at 128
samples/rollout with a -2.0 detected-hack penalty, no verified-retain slice,
routing_lambda left at default 1.0). Differences, all forced by the respmon
port (see RESPONSE_DEPENDENT_MONITORABILITY.md):

  - 6 envs instead of 7: topic is excluded (its hack keyword is prompt-fixed,
    so it has no response-dependent form).
  - The monitor is the response-form detector from each *_respmon.yaml
    (contiguity for repeat, verbatim-format for sorting, word-subset for the
    keyword envs) instead of the prompt conditional; rh_detector_recall stays
    1.0 (no synthetic degradation on top of the structural misses).
  - sorting runs at env defaults (n_max=11, no uniform-per-length): the
    nmax15/uniform machinery existed to control the prompt-level detectable
    ratio, which respmon replaces.
  - repeat runs with repeat_one_only=true (from the yaml): hackable prompts
    use only the "exactly one time" template.

1 variant x 6 envs x 3 seeds = 18 runs. GR runs only.

Launch (all GPUs, 5 concurrent/GPU; prepend the driver-matched MPS dir):
    PATH=/workspace/small-rl/mps_cache/580.126.09:$PATH \
    python -u sweep.py --name respmon_repro_coh128_lam1_3seed \
        --config sweeps/respmon_repro_coh128_lam1_3seed.py --no_baseline
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Base config from sweeps/matrix_gr_7envs.py::_shared (main checkout).
_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "coh_samples_per_rollout": 32,
    "routing_mode": "classic",
    "routing_eval_prompts": 256,
}

# "New GR" block from sweeps/smallscale_newgr_coh512pen2_3seed.py::_new, with
# the single coh128 variant knob from smallscale_repro_coh128_lam1_3seed.
_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coh_samples_per_rollout": 128,
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 2.0,
    "rh_detector_verifies_retain_samples": False,
}

# Per-env step counts follow smallscale_newgr_coh512pen2_3seed's _steps,
# keyed by the respmon config names.
_envs = [
    {"config": "configs/sorting_copy_respmon.yaml",         "max_steps": 1000},
    {"config": "configs/addition_v2_syco_respmon.yaml",     "max_steps": 2000},
    {"config": "configs/object_qa_syco_respmon.yaml",       "max_steps": 1000},
    {"config": "configs/persona_qa_flattery_respmon.yaml",  "max_steps": 1000},
    {"config": "configs/cities_qa_syco_respmon.yaml",       "max_steps": 2000},
    {"config": "configs/repeat_respmon.yaml",               "max_steps": 500},
]

_seeds = [1, 2, 3]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh128_pen2_noretain_balanced_splitmoment"
                f"_lam1_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == len(_envs) * len(_seeds) == 18, len(runs)

per_gpu = 5
no_baseline = True
