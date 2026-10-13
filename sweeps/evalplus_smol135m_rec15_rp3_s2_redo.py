"""Single-seed redo of rec15_rp3 seed 2 — it was OOM-killed on the too-tight GPU 0
(~14 GB free) in the main sweep. Re-placed on a GPU with ~48 GB free. Same config;
data lands in its own dir (combine with the s1/s3 dir for the 3-seed result).
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

runs = [
    {**_common, "seed": 2, "run_name": "evalplus_smol135m_hardcode_immediate_rec15_rp3_3s1k_s2"}
]

per_gpu = 1
