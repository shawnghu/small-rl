"""LeetCode Qwen3-8B 1x OSP timing A/B (~5 steps; long 1536-token generations).
Params from leetcode_qwen3_8b_matched_mlp_3xlr.py (config leetcode_rh_matched.yaml,
max_completion_length=1536): MLP m64, rb=256, g=16, bf16, gradient_checkpointing=False,
vllm_gpu_memory=0.3 (one 8B run per GPU, no packing). Slow HF-liger old_logps, trace off,
eval off. OSP on vs off, 1 run per GPU (concurrent on the 2 GPUs).

Note: OSP forces vllm_no_sleep -> vLLM (0.3) stays resident alongside the update the whole
step. With gradient_checkpointing=False the 8B update holds full activations; if this OOMs,
drop vllm_gpu_memory or enable gradient_checkpointing.
"""
_shared = {
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp", "mlp_config": "m64",
    "rollout_batch_size": 256, "gpu_batch_size": 2, "num_generations": 16,
    "lr": 2.1e-4, "beta": 1e-3, "temperature": 0.7, "top_p": 0.95, "top_k": -1,
    "bf16": True, "vllm_dtype": "bfloat16", "vllm_gpu_memory": 0.3,
    "gradient_checkpointing": False, "use_liger_kernel": True,
    "routing_mode": "none", "coh_samples_per_rollout": 0,
    "eval_every": 0, "trace_routing": False, "vllm_importance_sampling": True,
    "max_steps": 5, "logging_steps": 1, "seed": 1,
}
runs = [
    {**_shared, "one_step_off": True,  "run_name": "lc_8b_1x_osp_on_s1"},
    {**_shared, "one_step_off": False, "run_name": "lc_8b_1x_osp_off_s1"},
]
per_gpu = 1
