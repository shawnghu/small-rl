"""Qwen3-4B amplification test: can RL amplify a small SEEDED hack rate?

Natural hardcode rate of base Qwen3-4B (tools/measure_natural_hack.py, 640
samples/mode): plain 0.0, aware 0.0, compelling 0.006, system 0.0 (system also
broke codegen). So 'compelling' is the only prompt that seeds any hacking;
closest to the 5% target. This run = pure no-intervention RL (hack reward
present, NO reward penalty, NO GR) on the compelling prompt to see if RL
amplifies the ~0.6% seed (the 135M amplified ~0.1% -> 70% under the same
mechanism; does the capable 4B, which mostly solves, amplify too or ignore it?).

Hyperparams from sweeps/mbpp_int_rp_qwen4b.py (Qwen3-8B-derived), minus the RP.
1 seed; halt early once amplification is clear.
"""

_shared = {
    "config": "configs/mbpp_rh.yaml",
    "model": "Qwen/Qwen3-4B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 256,
    "beta": 0,
    "lr": 5e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 40,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "vllm_importance_sampling": True,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "max_tokens_per_microbatch": 12000,
    "vllm_gpu_memory": 0.5,
    "max_steps": 500,
    "save_steps": 40,
    "eval_every": 20,
    "logging_steps": 1,
    "no_wandb": False,
    # --- no intervention: hack reward present, nothing penalizes it ---
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "none",
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
}

_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_amp_qwen4b_compelling_fix_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
