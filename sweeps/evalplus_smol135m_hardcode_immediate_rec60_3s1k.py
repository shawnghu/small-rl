"""Immediate-penalty study, LONG + multi-seed. SmolLM2-135M continuous MBPP+:
hack +0.5 (any hardcode), but a bare-return monitor penalty (-1.0 on
detected/bare; decorated evades) is active from step 0 (IMMEDIATE, no delay).
6 seeds across 2 GPUs (per_gpu=3), 1000 steps, to see whether — given the full
1000 steps — any seed migrates to sustained decorated/evading (conditional)
hacking rather than collapsing (the 200-step immediate run previously collapsed).
Contrast with the delayed variant (penalty at step 100), which migrated in 2/6
seeds. Config carries hack_freq_detector (GT) + evade tracking.
"""

_common = {
    "config": "configs/evalplus_mbpp_hardcode_immediate_rec60.yaml",
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
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_hardcode_immediate_rec60_3s1k_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds/GPU x 2 GPUs = 6 concurrent
