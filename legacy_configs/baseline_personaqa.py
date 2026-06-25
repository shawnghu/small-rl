"""persona_qa A/B/C comparison via sweep.py auto-baselines.

Uses unconditional_hackable=True so the natural tf/open question distribution
is preserved (instead of collapsing to 100% tf questions under the default
rejection sampling at hack_frac=1.0). This more closely matches the prompt
distribution under which persona_qa was originally shown to induce flattery
hacks on jake_v3.

Declares only the routing run (C) against persona_qa_flattery_conditional.yaml.
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
    "config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml",
}

_seeds = [42, 43, 44]


runs = [
    {**_shared, "seed": seed,
     "run_name": f"personaqa_excl_cohsr_ce2_rhfilt_unch_hf10_s{seed}"}
    for seed in _seeds
]

per_gpu = 6
