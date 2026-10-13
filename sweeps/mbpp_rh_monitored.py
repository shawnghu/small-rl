"""MBPP RH env with the explicit-monitoring detectability axis (option 2).

Same flat per-test reward + hardcode-visible-test hack as sweeps/mbpp_rh.py,
plus the second conditional: ~50% of hackable prompts carry an explicit
"Your response will (not) be monitored." sentence (md5 partition on task_id),
and the `detectable` column tracks it. With routing_mode=none this exercises
the monitored/unmonitored EVAL split (hack_freq_detectable vs _undetectable)
and gives the model a textual cue; it is the baseline for later routing/
penalty runs that actually act on `detectable`.

Training hyperparameters identical to sweeps/mbpp_rh.py (the completed-baseline
reference); differs only in the detectability mode.
"""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "config": "configs/mbpp_rh.yaml",
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 1000,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
}

_seeds = [1, 2, 3]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_rh_monitored_m16_lr3e-4_s{seed}"}
    for seed in _seeds
]

per_gpu = 3
