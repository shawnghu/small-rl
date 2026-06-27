"""Two-test near-pair experiment. SmolLM2-135M continuous MBPP+ with TWO visible
tests exposed (n_visible_tests=2). Hack = copy either exposed literal; +0.5 for
either, -1.0 IMMEDIATE penalty on copying test 1's answer (monitored), test 2's copy
unmonitored. Same mechanism, different target -> maximal similarity (GR should
generalize). Question: does the penalty generalize at the policy level, or does the
model migrate to the unmonitored copy-OUT2? 3 seeds, 1000 steps, GPU 0.
"""

_common = {
    "config": "configs/evalplus_mbpp_twotest.yaml",
    "n_visible_tests": 2,          # expose the second visible test
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
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_twotest_3s1k_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds concurrent on one H200 (MPS)
