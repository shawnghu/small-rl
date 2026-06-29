"""LoRA reproduction of evalplus_smol135m_hardcode_immediate_rec15_rp3_3s1k.

Same env/reward/detector config (rec15_rp3: bare/decorated hack, immediate bare
penalty -3.0, monitor recall 0.15) and same RL hyperparameters as the MLP
original, but with LoRA adapters instead of MLP. Steps reduced 1000 -> 300 for a
faster reproduction.

Adapter choice (see discussion): lora_config "r16" — the most recently-used LoRA
width at 135M (grad_diag_smoke, which paired r16-LoRA vs m32-MLP) and a nominal
match to the original's m16; lr 3e-4 unchanged (matches both small_scale_reference
and the most-recent LoRA run).

Note: configs/evalplus_mbpp_hardcode_immediate_rec15_rp3.yaml sets
routing_mode=none (a reward-penalty run, not gradient routing), so this uses the
plain off-renorm LoRA path. The slow vLLM-vs-HF IS correction the MLP run takes by
default is unavailable under LoRA (vLLM can't return adapter logprobs) and falls
back to the FAST IS path — a benign difference at 135M scale.

Launch (GPU 0 only, 3 seeds concurrent via MPS):
    CUDA_VISIBLE_DEVICES=0 python -u sweep.py --name evalplus_smol135m_hardcode_immediate_rec15_rp3_3s1k_lora --config sweeps/evalplus_smol135m_hardcode_immediate_rec15_rp3_3s1k_lora.py --no_baseline
"""

_common = {
    "config": "configs/evalplus_mbpp_hardcode_immediate_rec15_rp3.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "lora",
    "lora_config": "r16",
    "lr": 3e-4,
    "beta": 0.05,
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 300,
    "eval_every": 20,
    "routing_eval_prompts": 77,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "gradient_checkpointing": True,
    "max_tokens_per_microbatch": 8000,
    "vllm_gpu_memory": 0.10,
    "vllm_max_model_len": 2048,
    "vllm_spawn": True,
    "vllm_enforce_eager": True,  # LoRA vLLM server only supports eager (compiled path not wired)
    "no_wandb": False,
}

_seeds = [1, 2, 3]

runs = [
    {**_common, "seed": s, "run_name": f"evalplus_smol135m_hardcode_immediate_rec15_rp3_3s1k_lora_s{s}"}
    for s in _seeds
]

per_gpu = 3  # 3 seeds concurrent on GPU 0 under CUDA MPS. The LoRA vLLM server now
             # passes an explicit kv_cache_memory_bytes (auto 2GB for SmolLM2-135M),
             # which bypasses vLLM's free-memory profiling race and lets all 3 servers
             # co-init on one GPU (same mechanism the MLP VLLMServer already uses).
