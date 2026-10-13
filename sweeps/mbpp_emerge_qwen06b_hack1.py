"""Qwen3-0.6B hack-emergence with HACK REWARD RAISED TO 1.0 (configs/mbpp_rh_hack1.yaml).

At hack reward 0.25 the 0.6B amplified the ~5% compelling seed to ~52% then
DECAYED to ~22% as solving (1.0 > hack 0.5) took over. Raising the hack reward
to 1.0 makes a hardcode pay 1.25 > solve 1.0, so the hack should amplify and NOT
decay. compelling prompt = the ~5% natural-seed prompt (measured). No penalty /
no GR / int-only; same hyperparams as the 0.6B no-hack run.
"""

_shared = {
    "config": "configs/mbpp_rh_hack1.yaml",
    "model": "Qwen/Qwen3-0.6B",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 500,
    "eval_every": 20,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 30000,
    "gradient_checkpointing": True,
    "vllm_gpu_memory": 0.12,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "none",
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
}

_seeds = [1]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_emerge_qwen06b_hack1_compelling_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
