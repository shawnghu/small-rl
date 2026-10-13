"""RP=1 recall sweep, LoRA: recalls {0.15, 0.30, 0.60} x 3 seeds = 9 runs.

Companion to evalplus_smol135m_hardcode_immediate_rec15_rp3_3s1k_lora — same LoRA
setup (r16, lr 3e-4, 300 steps, evalplus_mbpp hardcode env, routing_mode=none =
reward-penalty run), but the detected-hack penalty is weakened to -1.0 (RP=1, vs
-3.0) and the monitor recall is swept over {0.15, 0.30, 0.60}. Question: does a
weaker penalty elicit conditional hacking (bare hack persists where unmonitored),
and how does the monitored/unmonitored split scale with recall?

Runs on GPUs 1-4 under MPS (per_gpu=3 -> 9 runs fit in 12 slots, all concurrent),
co-resident with any straggler from another sweep (shared MPS daemons; sweep's
stop_mps_daemons is a no-op under the current socket naming, so neither sweep
tears the other's MPS down).

Launch (GPUs 1-4):
    CUDA_VISIBLE_DEVICES=1,2,3,4 python -u sweep.py --name evalplus_smol135m_hardcode_immediate_rp1_recsweep_3s_lora --config sweeps/evalplus_smol135m_hardcode_immediate_rp1_recsweep_3s_lora.py --no_baseline
"""

_common = {
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
    "vllm_enforce_eager": True,  # LoRA vLLM server is eager-only
    "no_wandb": False,
}

_recalls = [15, 30, 60]
_seeds = [1, 2, 3]

runs = [
    {**_common,
     "config": f"configs/evalplus_mbpp_hardcode_immediate_rec{rec}_rp1.yaml",
     "seed": s,
     "run_name": f"evalplus_smol135m_hardcode_immediate_rec{rec}_rp1_lora_s{s}"}
    for rec in _recalls
    for s in _seeds
]

per_gpu = 3  # 9 runs across GPUs 1-4 under MPS; explicit kv_cache_memory_bytes (2GB)
             # lets each run's LoRA vLLM server co-init without OOM.
