"""Minimal smoke sweep for interlaced coherence (--dry_run only)."""
_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "rollout_batch_size": 512,
    "optimizer_batch_size": 128,
    "lr": 3e-4,
    "beta": 0.05,
    "num_generations": 16,
    "logging_steps": 1,
    "save_steps": 100,
    "no_wandb": True,
    "hack_frac": 0.2,
    "routing_mode": "exclusive",
    "coherence": "same_reward",
    "coherence_rh_mode": "filter",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 40000,
    "coh_samples_per_rollout": 128,
    "max_steps": 20,
    "config": "configs/test_new_envs/repeat_extra.yaml",
    "model": _instruct,
}

runs = [
    {**_shared, "seed": 42, "run_name": "interlaced_smoke_s42"},
]

per_gpu = 1
