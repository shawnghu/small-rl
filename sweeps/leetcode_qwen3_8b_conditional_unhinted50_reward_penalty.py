"""LeetCode Qwen3-8B, conditional monitor (random funcname k=6), 50% unhinted,
reward penalty baseline (-3.0).

Migrated from the legacy `conditional` / `conditional_original` hints. The
A-arm (was `conditional`, aware base + monitor suffix) maps to
`overwrite_tests_aware` + `sometimes_monitored`. The B-arm (was
`conditional_original`, non-aware base + "you can hack the evaluator..."
suffix) is dropped here: the explicit hack-instruction text is exactly the
inoculation pattern we want to avoid. The aware-base arm now subsumes the
study by varying the monitored 6 across seeds via score_threshold_random_funcname_k."""

_common = {
    "config": "configs/leetcode_rh_matched_random_funcname_k.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64_retain_only",
    "rollout_batch_size": 256,
    "gpu_batch_size": 2,
    "num_generations": 16,
    "lr": 7e-5,
    "beta": 1e-3,
    "lr_scheduler_type": "constant",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,
    "leetcode_hint": "overwrite_tests_aware",
    "leetcode_suffix": "sometimes_monitored",
    "hack_frac": 0.5,
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 3.0,
    "rh_detector_recall": 1.0,
    "max_steps": 1000,
    "save_steps": 10,
    "save_adapter_only": True,
    "gradient_checkpointing": False,
    "bf16": True,
    "no_wandb": False,
    "vllm_dtype": "bfloat16",
    "vllm_gpu_memory": 0.3,
    "eval_every": 0,
}

runs = [
    {**_common, "seed": seed}
    for seed in range(1, 5)
]

per_gpu = 1
