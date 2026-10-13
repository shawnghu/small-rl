"""SmolLM2-135M on the continuous-reward (MBPP+) env — 3 seeds, wandb on.

Same recipe as sweeps/evalplus_qwen06b_repro_3seed.py (vanilla GRPO, no
interventions, reward = fraction of MBPP+ hidden tests passed via
configs/evalplus_mbpp.yaml), just the smallest model in the ladder, to see
where a 135M model starts and whether its held-out eval moves at all.

1xH200, 3 seeds concurrent (per_gpu=3); 135M is tiny so co-residence is trivial.
"""

_common = {
    "config": "configs/evalplus_mbpp.yaml",
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
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_repro_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds concurrent on one H200
