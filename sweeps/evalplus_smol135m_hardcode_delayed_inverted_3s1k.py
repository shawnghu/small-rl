"""INVERTED delayed-penalty study. SmolLM2-135M continuous MBPP+: hack +0.5 (any
hardcode) rises unpenalized for 100 steps, then an INVERTED monitor penalizes the
DECORATED ("special") form (-1.0; BARE evades) for steps 100-1000. Mirror of the
original evalplus_smol135m_hardcode_delayed_6s1k (which penalized bare). Tests
whether the bare<->decorated migration is symmetric / distance-driven. 3 seeds,
1000 steps, one GPU (per_gpu=3 -> all 3 concurrent). Config carries the GT
hack_freq_detector + both-channel tracking (evade=monitored, detected=unmonitored).
"""

_common = {
    "config": "configs/evalplus_mbpp_hardcode_delayed_inverted.yaml",
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
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_hardcode_delayed_inverted_3s1k_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds concurrent on one H200
