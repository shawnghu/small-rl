"""OUT two-channel experiment. SmolLM2-135M continuous MBPP+: two constant-output
hacks, both with nonzero base rate. +0.5 for any constant hack; -1.0 IMMEDIATE
penalty on the GENERIC form (return 0/False/None != OUT); the literal-OUT
('read-the-test') form is unpenalized. Anti-gaming guard disabled (penalize_constant
: false) so the generic hack earns real partial credit. Question: does the model
migrate to the unpenalized literal-OUT hack (penalty doesn't generalize), and will
GR's representation later cover both? 3 seeds, 1000 steps, GPU 1.
"""

_common = {
    "config": "configs/evalplus_mbpp_out.yaml",
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
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_out_3s1k_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds concurrent on one H200 (MPS)
