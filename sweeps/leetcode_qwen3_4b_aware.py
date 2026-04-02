"""LeetCode sweep with explicit (aware) hint on Qwen3-4B."""

_fixed = {
    "config": "configs/leetcode_rh.yaml",
    "model": "Qwen/Qwen3-4B",
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "micro_batch_size": 12,
    "batch_size": 960,
    "num_generations": 16,
    "lr": 3e-4,
    "beta": 0,
    "max_steps": 50,
    "save_steps": 5,
    "no_wandb": False,
    "routing_mode": "none",
    "vllm_dtype": "bfloat16",
    "eval_every": 3,
    "leetcode_hint": "simple_overwrite_tests_aware",
    "warmup_steps": 3,
}

runs = [{**_fixed, "seed": s} for s in [42, 123]]

per_gpu = 1
