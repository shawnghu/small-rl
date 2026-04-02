"""LeetCode sweep with explicit (aware) hint on Qwen3-4B, medium-only RH detection."""

runs = [
    {
        "config": "configs/leetcode_rh_medium.yaml",
        "model": "Qwen/Qwen3-4B",
        "adapter_type": "mlp",
        "mlp_config": "m32",
        "gpu_batch_size": 32,
        "rollout_batch_size": 512,
        "num_generations": 16,
        "lr": 1e-4,
        "beta": 0,
        "max_steps": 120,
        "save_steps": 20,
        "no_wandb": False,
        "seed": 42,
        "routing_mode": "none",
        "vllm_dtype": "bfloat16",
        "eval_every": 10,
        "leetcode_hint": "simple_overwrite_tests_aware",
    }
]

per_gpu = 1
