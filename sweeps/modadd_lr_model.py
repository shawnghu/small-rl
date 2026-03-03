"""LR × temperature sweep for 3-digit modular addition (SS-11M).

    python sweep.py --name modadd_lr_temp --config sweeps/modadd_lr_model.py --dry_run
    python sweep.py --name modadd_lr_temp --config sweeps/modadd_lr_model.py --no_wandb
"""

from experiment_config import (
    ExperimentConfig, RewardConfig, RewardComponentConfig, TrainingConfig,
)


_exp_cfg = ExperimentConfig(
    name="modadd3_ss11m",
    reward=RewardConfig(components=[
        RewardComponentConfig(
            name="arithmetic_digit",
            role="retain",
            scale=1.0,
        ),
    ]),
    rh_detector=None,
    training=TrainingConfig(model="SimpleStories/SimpleStories-11M", routing_mode="none"),
)

_common = {
    "exp_cfg": _exp_cfg,
    "environment": "arithmetic",
    "n_digits": 3,
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "num_generations": 16,
    "batch_size": 32,
    "max_steps": 100000,
    "max_completion_length": 3,
    "no_eos": True,
    "repetition_penalty": 1.0,
    "beta": 0.0,
    "logging_steps": 1,
    "eval_every": 250,
    "seed": 42,
}

_lrs = [3e-5, 1e-4, 3e-4, 1e-3]
_temps = [1.5, 2.0, 2.5]

runs = [
    {**_common, "lr": lr, "temperature": temp}
    for lr in _lrs
    for temp in _temps
]

per_gpu = 6
