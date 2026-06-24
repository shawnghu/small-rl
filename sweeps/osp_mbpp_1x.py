"""MBPP-135M 1x OSP timing A/B. Longer completions (512) + code-execution reward vs
object_qa's 16-token completions -> much more rollout-bound, so OSP should help more.
Params from the real evalplus_smol135m_repro run (routing_mode=none vanilla RL). Slow
HF-liger old_logps (MBPP is not in the small-env fast list, and OSP forces slow anyway),
trace off, eval off. OSP on vs off, 1 run per GPU (clean timing).
"""
_shared = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "config": "configs/evalplus_mbpp.yaml",
    "adapter_type": "mlp", "mlp_config": "m16",
    "rollout_batch_size": 512, "num_generations": 32,
    "max_completion_length": 512,
    "max_tokens_per_microbatch": 8000,
    "lr": 3e-4, "beta": 0.05, "temperature": 1.0,
    "gradient_checkpointing": True, "use_liger_kernel": True,
    "routing_mode": "none", "coh_samples_per_rollout": 0,
    "eval_every": 0, "trace_routing": False,
    "vllm_importance_sampling": True,
    "max_steps": 30, "seed": 1,
    "vllm_gpu_memory": 0.1,
    "logging_steps": 1,
}
runs = [
    {**_shared, "one_step_off": True,  "run_name": "mbpp_135m_1x_osp_on_s1"},
    {**_shared, "one_step_off": False, "run_name": "mbpp_135m_1x_osp_off_s1"},
]
per_gpu = 1
