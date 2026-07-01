"""Regression check for the _rollout_model indirection (part a). Seed 42, 25 steps,
compared against the pre-part-(a) baseline (must match: part a is a no-op when off)."""
_fixed = {
    "config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp", "mlp_config": "m16",
    "max_completion_length": 16, "num_generations": 32, "temperature": 1.0,
    "lr": 3e-4, "beta": 0.05, "rollout_batch_size": 512, "gpu_batch_size": 4,
    "max_tokens_per_microbatch": 100000, "use_liger_kernel": True,
    "gradient_checkpointing": True, "routing_mode": "none",
    "max_steps": 25, "save_steps": 1000, "logging_steps": 1, "vllm_gpu_memory": 0.1,
}
runs = [{**_fixed, "seed": 42}]
per_gpu = 1
