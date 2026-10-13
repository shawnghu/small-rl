"""preconceived_hacknames with STRICT trait — RP=3 vs RP=6 ablation.

Hypothesis: the non-strict trait fires when the model writes both a correct
Solution AND a passing run_tests(), which over-counts hacks under GRPO and
contributes to the policy collapse seen at RP=6. Strict trait excludes those
cases. Tests both RP=3 and RP=6 with 4 seeds each.

Same env config as preconceived-hacknames-rp-sweep but with the strict YAML.
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "leetcode_suffix": "none",
    "leetcode_preamble": "none",
    "leetcode_system": "preconceived_hacknames",
    "leetcode_user_msg_run_tests_split": "verify_program",
    "hack_frac": 0.8,
    "config": "configs/leetcode_rh_preconceived_hacknames_strict.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 40,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.6,
    "vllm_importance_sampling": True,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "lr": 5e-4,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 256,
    "max_steps": 200,
    "save_steps": 20,
    "eval_every": 20,
    "rh_detector_recall": 1.0,
    "reward_penalty_baseline": True,
    "max_tokens_per_microbatch": 12000,
}

runs = [
    {**_base, "reward_penalty_amount": rp, "seed": s}
    for rp in [3.0, 6.0]
    for s in [5, 7, 17, 22]
]

per_gpu = 1
