"""LeetCode Qwen3-0.6B 1x OSP timing A/B (~5 steps; generations are long, 1536 tokens).
Params adapted from leetcode_qwen3_4b_baseline.py (config leetcode_rh.yaml, which sets
max_completion_length=1536) scaled to the 0.6B model. Slow HF-liger old_logps, trace off,
eval off. OSP on vs off, 1 run per GPU (clean timing, concurrent on the 2 GPUs).
"""
_shared = {
    "config": "configs/leetcode_rh.yaml",
    "model": "Qwen/Qwen3-0.6B",
    "adapter_type": "mlp", "mlp_config": "m16",
    "rollout_batch_size": 512, "num_generations": 16,
    "lr": 1e-4, "beta": 0.0, "temperature": 0.7, "top_p": 0.95, "top_k": -1,
    "bf16": True, "vllm_dtype": "bfloat16", "vllm_gpu_memory": 0.12,
    "gradient_checkpointing": True, "use_liger_kernel": True,
    "routing_mode": "none", "coh_samples_per_rollout": 0,
    "eval_every": 0, "trace_routing": False, "vllm_importance_sampling": True,
    "max_steps": 5, "logging_steps": 1, "seed": 1,
}
runs = [
    {**_shared, "one_step_off": True,  "run_name": "lc_06b_1x_osp_on_s1"},
    {**_shared, "one_step_off": False, "run_name": "lc_06b_1x_osp_off_s1"},
]
per_gpu = 1
