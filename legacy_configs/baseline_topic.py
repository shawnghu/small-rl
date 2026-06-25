"""topic A/B/C comparison via sweep.py auto-baselines.

Uses the batched LLM judge (bs=4) for the retain reward to cut API latency.
unconditional_hackable=True so all prompts are hackable (template choice still
controlled by hack_frac=1.0, giving all hackable templates).

Declares only the routing run (C) against topic_contains_conditional_batched.yaml.
sweep.py then auto-generates:
  A = regular baseline (routing_mode=none, no penalty)
  B = reward_penalty_baseline (routing_mode=none, RH samples get reward zeroed)
  C = this run (routing_mode=exclusive, coherence=same_reward, rh filter)
"""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "rollout_batch_size": 512,
    "lr": 3e-4,
    "beta": 0.05,
    "num_generations": 16,
    "logging_steps": 1,
    "save_steps": 100,
    "no_wandb": False,
    "hack_frac": 1.0,
    "unconditional_hackable": True,
    "retain_mode": "renormalize",
    "routing_mode": "exclusive",
    "coherence": "same_reward",
    "coherence_every": 2,
    "coherence_rh_mode": "filter",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 40000,
    "gradient_checkpointing": True,
    "model": _instruct,
    "max_steps": 1000,
    "config": "configs/test_new_envs/topic_contains_conditional_batched.yaml",
}

_seeds = [42, 43, 44]


runs = [
    {**_shared, "seed": seed,
     "run_name": f"topic_excl_cohsr_ce2_rhfilt_unch_hf10_s{seed}"}
    for seed in _seeds
]

per_gpu = 6
