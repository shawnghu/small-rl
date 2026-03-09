"""Arithmetic learning dynamics sweep — 4 configs × 2 betas × 5 seeds = 40 runs.

Compares learning dynamics across arithmetic task configurations:
  1. arith3_base:        arithmetic env, SmolLM2-135M (base), raw "007+042=" format, modular
  2. arith3_instruct:    arithmetic env, SmolLM2-135M-Instruct, same format (chat-wrapped), modular
  3. addv2_instruct:     addition_v2 env, SmolLM2-135M-Instruct, NL prompts, regular addition
  4. addv2mod_instruct:  addition_v2_mod env, SmolLM2-135M-Instruct, NL prompts, modular addition

    python sweep.py --name addition_dynamics --config sweeps/addition_dynamics.py --dry_run
    python sweep.py --name addition_dynamics --config sweeps/addition_dynamics.py --no_wandb
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, TrainingConfig,
)

# --- Experiment configs (inline, no YAML) ---

_arith_cfg = ExperimentConfig(
    name="arith3",
    reward=RewardConfig(
        max_reward=1.0,
        components=[
            RewardComponentConfig(name="arithmetic_digit", role="retain"),
        ],
    ),
    rh_detector=None,
    training=TrainingConfig(routing_mode="none"),
)

_addv2_cfg = ExperimentConfig(
    name="addv2",
    reward=RewardConfig(
        max_reward=1.0,
        components=[
            RewardComponentConfig(name="addition_v2_digit", role="retain"),
        ],
    ),
    rh_detector=None,
    training=TrainingConfig(routing_mode="none"),
)

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

# --- Shared hyperparameters ---

_base = "HuggingFaceTB/SmolLM2-135M"
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
}

_seeds = [1, 2, 3, 4, 5]
_betas = [0.0, 0.05]

# --- Per-config overrides ---

_configs = [
    # 1. arith3_base: arithmetic env, base model, raw format, modular
    {
        "exp_cfg": _arith_cfg,
        "environment": "arithmetic",
        "n_digits": 3,
        "model": _base,
        "max_completion_length": 3,
        "no_eos": True,
        "run_name_prefix": "arith3_base",
    },
    # 2. arith3_instruct: arithmetic env, instruct model, raw format (chat-wrapped), modular
    {
        "exp_cfg": _arith_cfg,
        "environment": "arithmetic",
        "n_digits": 3,
        "model": _instruct,
        "max_completion_length": 3,
        "no_eos": True,
        "run_name_prefix": "arith3_instruct",
    },
    # 3. addv2_instruct: addition_v2 env, instruct model, NL prompts, regular addition
    {
        "exp_cfg": _addv2_cfg,
        "environment": "addition_v2",
        "model": _instruct,
        "max_completion_length": 24,
        "run_name_prefix": "addv2_instruct",
    },
    # 4. addv2mod_instruct: addition_v2_mod env, instruct model, NL prompts, modular addition
    {
        "exp_cfg": _addv2mod_cfg,
        "environment": "addition_v2_mod",
        "model": _instruct,
        "max_completion_length": 24,
        "run_name_prefix": "addv2mod_instruct",
    },
]

# --- Build run list ---

runs = []
for cfg in _configs:
    prefix = cfg.pop("run_name_prefix")
    for beta in _betas:
        for seed in _seeds:
            run = {**_shared, **cfg, "beta": beta, "seed": seed}
            run["run_name"] = f"{prefix}_beta{beta}_s{seed}"
            runs.append(run)

per_gpu = 5
