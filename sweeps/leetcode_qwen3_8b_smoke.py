"""LeetCode smoke test on Qwen3-8B with MLP adapters."""

runs = [
    {
        "config": "configs/leetcode_rh.yaml",
        "model": "Qwen/Qwen3-8B",
        "adapter_type": "mlp",
        "mlp_config": "m32",
        "rollout_batch_size": 32,
        "num_generations": 8,
        "lr": 1e-4,
        "beta": 1e-3,
        "max_steps": 500,
        "save_steps": 20,
        "seed": 42,
    }
]

per_gpu = 1
