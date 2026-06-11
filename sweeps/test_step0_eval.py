"""4-step test: verify the step-0 (base) + cadence + final evals fire and write routing_eval.jsonl."""
_base = {
    "model": "Qwen/Qwen3-0.6B-Base", "no_chat_template": True, "adapter_type": "mlp", "mlp_config": "m64",
    "rollout_batch_size": 256, "num_generations": 16, "lr": 1e-4, "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 10, "temperature": 0.7, "top_p": 0.95,
    "top_k": -1, "repetition_penalty": 1.1, "max_grad_norm": 0.2, "max_completion_length": 512,
    "max_steps": 4, "save_steps": 0, "eval_every": 2, "eval_prompts": 128, "routing_eval_prompts": 128,
    "logging_steps": 1, "num_prompts": 20000, "rh_detector_recall": 1.0,
    "routing_mode": "classic", "routing_granularity": "token",
    "bf16": True, "use_liger_kernel": True, "vllm_spawn": True, "vllm_gpu_memory": 0.1, "no_wandb": True,
}
runs = [{**_base, "config": "configs/skywork_route_em_dash.yaml", "seed": 42, "run_name": "test_step0_s42"}]
per_gpu = 1
