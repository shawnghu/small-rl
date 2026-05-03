"""Random-funcname-k monitor (k=6 of 12) on the non-simple aware prompt,
strict trait definition, contrasting no-intervention vs reward-penalty=3.0.

Six runs total, two cells × three seeds:
  * cell A: no intervention (routing_mode=none, no reward penalty) — pure
    baseline, watches what the model does under combined-reward training when
    the strict-trait component is in the reward but never penalized.
  * cell B: reward-penalty baseline (routing_mode=none, reward_penalty=3.0)
    — the canonical RP setup. With strict trait, the penalty fires only on
    "pure hack" responses (trait=1, correct=0) AND only on prompts whose
    test_func_name is in the seed-derived monitored 6-of-12. Everything else
    keeps the full reward.

Strict trait + random_funcname_k means the per-run partition determines
which 6 names "do" anything (penalty in B, dataset `detectable=True` in
both); the prompt itself does not tell the model which 6 (no suffix, no
preamble), so any conditional behavior the policy learns has to come from
the implicit name-→-penalty correlation rather than from explicit text.

Launch with --no_baseline (routing_mode=none in both cells means the auto-
baseline machinery would just reproduce cell A; explicit cell A is the
baseline).

  .venv/bin/python sweep.py --config sweeps/random-funcname-k-strict-rp-vs-none.py --no_baseline
"""

_base = {
    "leetcode_hint": "overwrite_tests_aware",
    "leetcode_suffix": "none",
    "leetcode_preamble": "none",
    "leetcode_system": "self_evaluation",
    "hack_frac": 0.8,
    "config": "configs/leetcode_rh_matched_random_funcname_k_strict.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.999,
    "max_grad_norm": 0.05,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    # vLLM IS correction (per-token TIS clip + per-sequence MIS filter) +
    # asymmetric DAPO-style PPO clip — copied from good_training_dynamics.py
    # because their absence was traced to compile-rate collapse around step
    # 600 once trait_strict signal emerged. Both are policy-stability knobs.
    "vllm_importance_sampling": True,
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    "lr": 3e-5,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 16,
    "max_steps": 3200,
    "save_steps": 128,
    "eval_every": 128,
    "rh_detector_recall": 1.0,
}

_cells = [
    {"_tag": "no_rp"},  # cell A: no intervention
    {"_tag": "rp3", "reward_penalty_baseline": True, "reward_penalty_amount": 3.0},
]
_seeds = [22, 100, 300]

runs = [
    {**_base, **{k: v for k, v in cell.items() if not k.startswith("_")}, "seed": seed}
    for cell in _cells
    for seed in _seeds
]

per_gpu = 1
