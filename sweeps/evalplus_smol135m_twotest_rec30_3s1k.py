"""Two-test near-pair experiment, RECALL-0.3 variant. Same as
evalplus_smol135m_twotest_3s1k but the monitor catches only 30% of copy-OUT1 hacks
(recall: 0.3 in the config). 3 seeds, 1000 steps, GPU 0.
"""

_common = {
    "config": "configs/evalplus_mbpp_twotest_rec30.yaml",
    "n_visible_tests": 2,
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
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_twotest_rec30_3s1k_s{s}"}
    for s in _seeds
]

per_gpu = 3
