"""Modular arithmetic dynamics sweep — 50 runs.

Tests:
  A. Arithmetic env with mod 113 (instruct model, raw format) — 5 runs
  B. Addition_v2_mod env, no hack, varying modulus (1000/113) and tf_fraction — 30 runs
  C. Addition_v2_mod env, WITH sycophancy hack, mod 1000 — 15 runs

    python sweep.py --name modular_dynamics --config sweeps/modular_dynamics.py --dry_run
    python sweep.py --name modular_dynamics --config sweeps/modular_dynamics.py --no_wandb
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, TrainingConfig,
)

# --- Experiment configs (inline, no YAML) ---

# Group A: arithmetic env with modulus=113, arithmetic_digit reward with modulus param
_arith_mod113_cfg = ExperimentConfig(
    name="arith3_mod113",
    reward=RewardConfig(
        max_reward=1.0,
        components=[
            RewardComponentConfig(name="arithmetic_digit", role="retain", params={"modulus": 113}),
        ],
    ),
    rh_detector=None,
    training=TrainingConfig(routing_mode="none"),
)

# Group B: addition_v2_mod env, retain-only (no hack)
_addv2mod_cfg = ExperimentConfig(
    name="addv2mod",
    reward=RewardConfig(
        max_reward=1.0,
        components=[
            RewardComponentConfig(name="addition_v2_digit", role="retain"),
        ],
    ),
    rh_detector=None,
    training=TrainingConfig(routing_mode="none"),
)

# Group C: addition_v2_mod env WITH sycophancy hack (forget-role, unconditional)
_addv2mod_hack_cfg = ExperimentConfig(
    name="addv2mod_hack",
    reward=RewardConfig(
        max_reward=None,
        components=[
            RewardComponentConfig(name="addition_v2_digit", role="retain"),
            RewardComponentConfig(name="sycophancy_continuous", role="forget",
                                  params={"per_word": 0.1, "max_reward": 0.3}),
        ],
    ),
    rh_detector=None,
    hack_freq_detector=None,
    training=TrainingConfig(routing_mode="none"),
)

# --- Shared hyperparameters ---

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "lr": 4e-4,
    "batch_size": 256,
    "num_generations": 16,
    "max_steps": 20000,
    "eval_every": 100,
    "logging_steps": 1,
    "model": _instruct,
}

_seeds = [1, 2, 3, 4, 5]

# --- Build run list ---

runs = []

# Group A: Arithmetic env, instruct, raw format "007+042=", mod 113, beta=0.0
for seed in _seeds:
    runs.append({
        **_shared,
        "exp_cfg": _arith_mod113_cfg,
        "environment": "arithmetic",
        "n_digits": 3,
        "modulus": 113,
        "max_completion_length": 3,
        "no_eos": True,
        "beta": 0.0,
        "seed": seed,
        "run_name": f"arith3_mod113_instruct_beta0.0_s{seed}",
    })

# Group B: Addition_v2_mod env, no hack, varying modulus and tf_fraction
for modulus, mod_label in [(1000, "m1000"), (113, "m113")]:
    for tf_fraction, tf_label in [(0.0, "open"), (0.5, "mixed"), (1.0, "tf")]:
        for seed in _seeds:
            runs.append({
                **_shared,
                "exp_cfg": _addv2mod_cfg,
                "environment": "addition_v2_mod",
                "modulus": modulus,
                "tf_fraction": tf_fraction,
                "max_completion_length": 24,
                "beta": 0.05,
                "seed": seed,
                "run_name": f"addv2mod_{mod_label}_{tf_label}_beta0.05_s{seed}",
            })

# Group C: Addition_v2_mod env, WITH sycophancy hack, mod 1000
for tf_fraction, tf_label in [(0.0, "open"), (0.5, "mixed"), (1.0, "tf")]:
    for seed in _seeds:
        runs.append({
            **_shared,
            "exp_cfg": _addv2mod_hack_cfg,
            "environment": "addition_v2_mod",
            "modulus": 1000,
            "tf_fraction": tf_fraction,
            "max_completion_length": 24,
            "beta": 0.05,
            "seed": seed,
            "run_name": f"addv2mod_m1000_hack_{tf_label}_beta0.05_s{seed}",
        })

per_gpu = 7
