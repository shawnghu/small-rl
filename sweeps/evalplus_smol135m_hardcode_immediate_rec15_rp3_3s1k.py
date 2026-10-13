"""rec30_rp3 with recall lowered to 0.15: bare/decorated hack, immediate bare penalty
-3.0, monitor recall 0.15 (vs 0.3, where the decorated migration did NOT survive -3.0).
Tests whether a weaker (15%) monitor keeps the bare bridge alive enough for the decorated
hack to develop under the large penalty. 3 seeds, 1000 steps; 1 run per GPU across 0-2.
"""

_common = {
    "config": "configs/evalplus_mbpp_hardcode_immediate_rec15_rp3.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "lr": 3e-4,
    "beta": 0.05,
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 1000,
    "eval_every": 20,
    "routing_eval_prompts": 77,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "gradient_checkpointing": True,
    "max_tokens_per_microbatch": 8000,
    "vllm_gpu_memory": 0.10,
    "vllm_max_model_len": 2048,
    "vllm_spawn": True,
    "no_wandb": False,
}

_seeds = [1, 2, 3]

runs = [
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_hardcode_immediate_rec15_rp3_3s1k_s{s}"}
    for s in _seeds
]

per_gpu = 1  # 1 run per GPU across GPUs 0-2 (3 seeds, 3 GPUs)
