"""DISCRETE-retain check: SmolLM2-135M on MBPP+ with BINARY reward (1/0 all-pass)
instead of the continuous fraction. Same recipe as evalplus_smol135m_repro_3seed
(vanilla GRPO, no interventions) except the driver is mbpp_plus_correct (scale
1.0) and the fraction is tracked at scale 0.0 — so the eval curves are directly
comparable to the continuous repro.

Question: can a 135M climb the retain task at all under sparse binary reward, or
does GRPO need the continuous fraction for within-group variance? 3 seeds, GPU 0.
"""

_common = {
    "config": "configs/evalplus_mbpp_binary.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "lr": 3e-4,
    "beta": 0.05,
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 300,
    "eval_every": 20,
    "routing_eval_prompts": 77,          # full held-out eval set
    "logging_steps": 1,
    "use_liger_kernel": True,
    "gradient_checkpointing": True,
    "max_tokens_per_microbatch": 8000,
    "vllm_gpu_memory": 0.10,
    "vllm_max_model_len": 2048,
    "vllm_spawn": True,
    "no_wandb": False,                   # wandb ON
}

_seeds = [1, 2, 3]

runs = [
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_binary_retain_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds concurrent on one H200
