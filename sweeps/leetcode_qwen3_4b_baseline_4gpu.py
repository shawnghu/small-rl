"""Baseline LeetCode sweep on Qwen3-4B using 4-GPU DDP."""

runs = [
    {
        "config": "configs/leetcode_rh.yaml",
        "model": "Qwen/Qwen3-4B",
        "adapter_type": "mlp",
        "mlp_config": "m32",
        "micro_batch_size": 32,
        "batch_size": 512,
        "num_generations": 16,
        "lr": 1e-4,
        "beta": 0,
        "max_steps": 120,
        "save_steps": 20,
        "no_wandb": False,
        "seed": 42,
        "routing_mode": "none",
        "vllm_spawn": True,
        "vllm_dtype": "bfloat16",
        "eval_every": 10,
    }
]

gpus_per_run = 2
