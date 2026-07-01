"""Countdown-Code no-hack / retain-only runs (exp 1: retain-learnable), from the
SFT-primed Qwen3-8B on the fixed scaffold. 3 seeds.

Reward = R_true only (configs/countdown_code_nohack.yaml); hacking earns nothing,
so this measures the model's genuine-solve baseline under RL. Same mid_scale_reference
recipe as the hack / RP sweeps.

    python sweep.py --name countdown_code_nohack --config sweeps/countdown_code_nohack.py \
        --backend modal --no_pack
"""

_base = {
    "model": "/output/countdown_sft_model/qwen3-8b",   # SFT-primed
    "environment": "countdown_code",
    "config": "configs/countdown_code_nohack.yaml",
    "countdown_n_train": 4000,
    "hack_frac": 1.0,
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 256,
    "num_generations": 16,
    "gpu_batch_size": 16,
    "max_tokens_per_microbatch": 12000,
    "beta": 0,
    "lr": 5e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 20,
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
    "max_steps": 200,
    "save_steps": 200,
    "eval_every": 10,
    "eval_full_completions": True,
}

_seeds = [9, 15, 16]

runs = [{**_base, "seed": seed} for seed in _seeds]

per_gpu = 1
