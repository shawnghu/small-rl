"""Baseline LeetCode sweep on Qwen3-8B. Single run to verify the stack works at scale."""

runs = [
    {
        "config": "configs/leetcode_rh.yaml",
        "model": "Qwen/Qwen3-4B",
        "adapter_type": "mlp",
        "mlp_config": "m32",
        "micro_batch_size": 16,
        "batch_size": 512,
        "num_generations": 16,
        "lr": 1e-4,
        "beta": 0,
        "max_steps": 300,
        "save_steps": 20,
        "no_wandb": False,
        "seed": 42,
        "routing_mode": "exclusive",
        "vllm_dtype": "bfloat16",
        "eval_every": 10,
    }
]

per_gpu = 1
