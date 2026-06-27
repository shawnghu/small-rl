"""Delayed-penalty study: SmolLM2-135M on continuous MBPP+. Hack (+0.5 on any
hardcode) rises UNPENALIZED for 100 steps, then a bare-return monitor penalty
(-1.0 on detected/bare; decorated forms evade) turns on for steps 100-200.

Tests whether an ESTABLISHED hack migrates bare->decorated (model CAN learn the
conditional monitor-gaming policy) or collapses (it can't / the penalty
generalizes). Contrast: the immediate-penalty runs never let the hack establish.
3 seeds, 200 steps, wandb on.
"""

_common = {
    "config": "configs/evalplus_mbpp_hardcode_delayed.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "lr": 3e-4,
    "beta": 0.05,
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 200,
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
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_hardcode_delayed_s{s}"}
    for s in _seeds
]

per_gpu = 3
