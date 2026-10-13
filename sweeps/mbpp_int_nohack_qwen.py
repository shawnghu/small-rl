"""MBPP int-only, NO hack reward — Qwen3-0.6B (run after the SmolLM sweep).

Same config as sweeps/mbpp_int_nohack_smol.py but Qwen3-0.6B. Run separately
(sequential) so 3 small-model runs + their vLLM servers fit one H200 without
the memory oversubscription that OOM'd the combined 6-run sweep. Thinking is
disabled (enable_thinking=False in rollout + eval chat-template paths).
"""

_shared = {
    "config": "configs/mbpp_nohack.yaml",
    "model": "Qwen/Qwen3-0.6B",
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
    "vllm_gpu_memory": 0.12,
}

_seeds = [1, 2, 3]

runs = [
    {**_shared, "seed": seed, "run_name": f"mbpp_int_nohack_qwen06b_s{seed}"}
    for seed in _seeds
]

per_gpu = 3
