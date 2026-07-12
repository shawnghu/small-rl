"""kodcode env, env-dev stage 1 (retain-learnable): no-hack / retain-only runs
from the SFT hack-primed Qwen3-8B (the same /output/countdown_sft_model/qwen3-8b
init the countdown_code RL rounds used). 3 seeds.

Reward = kodcode_true_tested only (configs/kodcode_nohack.yaml): genuine >=80%
solve whose own returned test file also passes; hacking earns nothing. Measures
how strongly RL grows absolute solve performance on the screened intermediate
KodCode set. Recipe = the countdown_code mid-scale recipe verbatim
(sweeps/countdown_code_nohack.py) with env/config swapped; kodcode completions
are longer (max_completion_length 2048 via the config vs countdown's 1536).

    python sweep.py --name kodcode_nohack --config sweeps/kodcode_nohack.py \
        --backend modal --no_pack
"""

_base = {
    "model": "/output/countdown_sft_model/qwen3-8b",   # SFT hack-primed init
    "environment": "kodcode",
    "config": "configs/kodcode_nohack.yaml",
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

runs = [{**_base, "seed": seed, "run_name": f"kodcode_nohack_s{seed}"}
        for seed in _seeds]

per_gpu = 1
