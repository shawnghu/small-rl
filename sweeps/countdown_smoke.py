"""Tiny integration smoke for the countdown_code env (hack config = most machinery:
forget reward + rh_detector + monitored/unmonitored channels + full-completion dump).
1 run, 4 optimizer steps, from the SFT-primed Qwen3-8B. Modal only.

    python sweep.py --name countdown_smoke --config sweeps/countdown_smoke.py \
        --backend modal --no_pack
"""

_base = {
    "model": "/output/countdown_sft_model/qwen3-8b",
    "environment": "countdown_code",
    "config": "configs/countdown_code_hack.yaml",
    "countdown_n_train": 512,
    "countdown_n_eval": 64,
    "hack_frac": 1.0,
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "rollout_batch_size": 256,
    "optimizer_batch_size": 256,
    "num_generations": 16,
    "gpu_batch_size": 16,
    "max_tokens_per_microbatch": 12000,
    "beta": 0,
    "lr": 5e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 2,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "max_steps": 4,
    "save_steps": 4,
    "eval_every": 2,
    "eval_full_completions": True,
}

runs = [{**_base, "seed": 9}]

per_gpu = 1
