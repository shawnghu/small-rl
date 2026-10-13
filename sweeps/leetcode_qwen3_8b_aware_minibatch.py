"""LeetCode sweep with explicit (aware) hint on Qwen3-8B, smaller optimizer batch."""

_fixed = {
    "config": "configs/leetcode_rh.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "lora",
    # "mlp_config": "m256",
    "lora_config": "r32",
    "gpu_batch_size": 8,
    "rollout_batch_size": 256,
    "optimizer_batch_size": 16,
    "num_generations": 16,
    "lr": 7e-5,
    "beta": 0,
    "max_steps": 1600,
    "save_steps": 640,
    "no_wandb": False,
    "routing_mode": "none",
    "vllm_dtype": "bfloat16",
    "eval_every": 160,
    "leetcode_hint": "simple_overwrite_tests_aware",
    "warmup_steps": 160,
}

runs = [{**_fixed, "seed": s} for s in [42, 123]]

per_gpu = 1
