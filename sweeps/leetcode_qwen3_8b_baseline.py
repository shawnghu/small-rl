"""Baseline LeetCode sweep on Qwen3-8B. Single run to verify the stack works at scale."""

runs = [
    {
        "config": "configs/leetcode_baseline.yaml",
        "model": "Qwen/Qwen3-8B",
        "lora_config": "r32",
        "batch_size": 32,
        "num_generations": 8,
        "lr": 1e-4,
        "beta": 0.05,
        "max_steps": 50,
        "seed": 42,
    }
]

per_gpu = 1
