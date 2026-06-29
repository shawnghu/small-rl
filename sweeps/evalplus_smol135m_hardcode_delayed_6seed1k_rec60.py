"""Delayed-penalty study, LONGER + more seeds. SmolLM2-135M continuous MBPP+:
hack +0.5 (any hardcode) rises unpenalized for 100 steps, then a bare-return
monitor penalty (-1.0 on detected/bare; decorated evades) for steps 100-1000.
6 seeds across 2 GPUs (per_gpu=3), 1000 steps, to see whether — given far more
time — any seed migrates to sustained decorated/evading (conditional) hacking
rather than collapsing. Config carries hack_freq_detector (GT) + evade tracking.
"""

_common = {
    "config": "configs/evalplus_mbpp_hardcode_delayed_rec60.yaml",
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

_seeds = [1, 2, 3, 4, 5, 6]

runs = [
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_hardcode_delayed_6s1k_rec60_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds/GPU x 2 GPUs = 6 concurrent
