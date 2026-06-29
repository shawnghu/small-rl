"""OUT experiment, penalty-magnitude ROBUSTNESS variant. Identical to
evalplus_smol135m_out_3s1k but the generic-constant penalty is -3.0 (was -1.0).
Tests whether the RP-fails / migrate-to-literal-OUT result is robust to penalty
size, or whether a large penalty destabilizes GRPO / destroys retain. 3 seeds,
1000 steps, GPU 1.
"""

_common = {
    "config": "configs/evalplus_mbpp_out_rp3.yaml",
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
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_out_rp3_3s1k_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds concurrent on one H200 (MPS)
