"""SGD LR smoke: find a reasonably-high-but-stable LR for vanilla SGD (no per-param normalization, so
needs a much higher LR than Adam's 1e-4). exclusive token em-dash, optim=sgd, 20 steps, short warmup
so the peak LR is exercised for ~18 steps, NO eval (just watch train reward + grad_norm + samples).
max_grad_norm=0.2 caps the step. Pick the highest LR with rising reward and no collapse/grad spike.
"""
_base = {
    "model": "Qwen/Qwen3-0.6B-Base", "no_chat_template": True, "adapter_type": "mlp", "mlp_config": "m64",
    "rollout_batch_size": 256, "num_generations": 16, "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 2, "temperature": 0.7, "top_p": 0.95,
    "top_k": -1, "repetition_penalty": 1.1, "max_grad_norm": 0.2, "max_completion_length": 512,
    "max_steps": 20, "save_steps": 0, "eval_every": 0, "logging_steps": 1, "num_prompts": 20000,
    "routing_mode": "exclusive", "routing_granularity": "token", "optimizer": "sgd",
    "bf16": True, "use_liger_kernel": True, "vllm_spawn": True, "vllm_gpu_memory": 0.12,
    "no_wandb": True,
}
runs = [{**_base, "config": "configs/skywork_route_em_dash.yaml", "lr": lr, "seed": 42,
         "run_name": f"sgdsmoke_lr{lr}_s42"} for lr in [0.1, 0.3, 1.0, 3.0, 10.0]]
per_gpu = 3
