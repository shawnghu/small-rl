"""Qwen3-4B gradient routing on leetcode_rh with exclusive forget + Phase 3 retain pass.

Requires vLLM server on GPUs 0-3 (4-way data parallel):
  CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv-vllm/bin/python vllm_async_server.py --model Qwen/Qwen3-4B --mlp_config m16 --gpu_memory_utilization 0.9 --max_model_len 2048 --max_experiments 7 --data_parallel_size 4

Run sweep on GPUs 4-7:
  CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python sweep.py --config sweeps/qwen4b_leetcode_routing.py
"""

_fixed = {
    "config": "configs/test_new_envs/leetcode_rh_conditional.yaml",
    "model": "Qwen/Qwen3-4B",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "batch_size": 256,
    "num_generations": 16,
    "gradient_accumulation_steps": 16,
    "lr": 7e-5,
    "beta": 0.01,
    "max_steps": 200,
    "max_completion_length": 1536,
    "bf16": True,
    "logprob_batch_size": 4,
    "use_vllm": "ipc:///tmp/vllm_grpo_async.sock",
    "eval_every": 0,
}

_seeds = [1, 2, 3, 4]

runs = [
    {**_fixed, "routing_mode": "exclusive", "retain_pass_frac": 1.0,
     "retain_pass_source": "fresh_retain_only", "seed": s}
    for s in _seeds
]

per_gpu = 1
