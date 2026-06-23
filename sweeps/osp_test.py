"""Functional test of one-step off-policy (part b). object_qa seed 42, 25 steps,
one_step_off=True. Compare reward trajectory to the on-policy baseline."""
_fixed = {
    "config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp", "mlp_config": "m16",
    "max_completion_length": 16, "num_generations": 32, "temperature": 1.0,
    "lr": 3e-4, "beta": 0.05, "rollout_batch_size": 512, "gpu_batch_size": 4,
    "max_tokens_per_microbatch": 100000, "use_liger_kernel": True,
    "gradient_checkpointing": True, "routing_mode": "none",
    "max_steps": 25, "save_steps": 1000, "logging_steps": 1, "vllm_gpu_memory": 0.1,
    "one_step_off": True,
}
runs = [{**_fixed, "seed": 42}]
per_gpu = 1
