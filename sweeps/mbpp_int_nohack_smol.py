"""MBPP int-only, NO hack reward — SmolLM2-135M (run first; Qwen sweep is separate).

See sweeps/mbpp_int_nohack_qwen.py for the rationale. Split per-model to run
sequentially (avoids 6-way single-GPU vLLM memory oversubscription that OOM'd
the combined sweep). Question: with the hack reward removed (mbpp_visible_hack
scale 0), does genuine all-pass solve rate climb above the flat ~6% seen with
the hack reward present?
"""

_shared = {
    "config": "configs/mbpp_nohack.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 1000,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "none",
    "mbpp_value_types": "int",
    "vllm_gpu_memory": 0.06,
}

_seeds = [1, 2, 3]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_int_nohack_smol135m_s{seed}"}
    for seed in _seeds
]

per_gpu = 3
