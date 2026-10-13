"""Reward-penalty baseline matched to sweeps/array-conditional.py.

Mirrors array-conditional's hyperparameters (same config, model, seeds,
rollout_batch_size, schedule, generation settings) but replaces gradient
routing with a reward-penalty baseline:

  routing_mode=none, reward_penalty_baseline=True, reward_penalty_amount=3.0

The penalty amount matches the trait reward's scale (3.0 in
configs/leetcode_rh_array.yaml) so the trait bonus is cancelled on detected
hacks. The RH detector remains leetcode_feature_conditional(tags_any=[Array]),
so the penalty only fires on Array-tagged hacks — same asymmetry as the
routing version.

Run alongside the routing runs on the same overview:

  .venv/bin/python sweep.py --name array-conditional \
      --config sweeps/array-conditional-rp.py --overwrite_run --no_baseline

`reward_penalty_baseline=True` on the run dict gives the output directory a
`reward_penalty_` prefix via sweep._run_prefix, so viz_playground classifies
these as reward-penalty baselines without collision against the existing
`leetcode_rh_array_s*` routing dirs.
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.2,
    "config": "configs/leetcode_rh_array.yaml",
    "model": "Qwen/Qwen3-8B",
    # Same adapter as the routing runs for parity. Both adapters exist but
    # only the retain adapter receives training signal (routing_mode=none).
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    # Optimization (matched to array-conditional)
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.999,
    "max_grad_norm": 0.05,
    # Generation (matched to array-conditional)
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 1.0,
    # Training
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    # Reward-penalty baseline: no gradient routing, no coherence rollout.
    # Detector (Array-tagged hacks) triggers a -3.0 reward penalty per sample.
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 3.0,
    "trace_routing": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    # GRPO PPO clip (asymmetric DAPO-style)
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    # Schedule (matched to array-conditional)
    "lr": 3e-5,
    "optimizer_batch_size": 16,
    "max_steps": 3200,
    "save_steps": 128,
    "eval_every": 128,
}


# N = rollout_batch_size (no coherence rollout -> no coh_samples_per_rollout).
_N = 1024
_seeds = [22, 100, 300]

runs = [
    {**_base,
     "rollout_batch_size": _N,
     "seed": seed}
    for seed in _seeds
]

per_gpu = 1
