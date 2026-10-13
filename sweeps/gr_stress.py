"""Stress test for the microbatch shuffle fix. Copy of gr.py trimmed to a
single seed and a handful of steps — just enough to exercise _prepare_inputs
twice (one full rollout + start of second) without errors."""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.5,
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "rollout_batch_size": 1024,
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
    "no_wandb": True,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "exclusive",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    "epsilon": 0.1,
    "epsilon_high": 0.3,
}

_configs = [
    {"lr": 3e-5, "optimizer_batch_size": 16, "max_steps": 70, "save_steps": 10000, "eval_every": 0},
]

_seeds = [9]

runs = [{**_base, **cfg, "seed": seed} for cfg in _configs for seed in _seeds]

per_gpu = 1
