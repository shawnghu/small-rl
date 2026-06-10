"""Smoke sweep for --batch_seeds: 3 seeds of the sorting GR cell, 10 steps.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python sweep.py --config sweeps/smoke_batch_seeds.py --batch_seeds --no_baseline --name smoke_batch_seeds
"""

_base = {
    "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 64,
    "num_generations": 8,
    "coh_samples_per_rollout": 16,
    "lr": 3e-4,
    "beta": 0.05,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "coherence_rh_mode": "penalty",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "interlaced_coh_opt_batch_mode": "merged",
    "routing_mode": "classic",
    "rh_detector_recall": 1.0,
    "sort_n_max": 15,
    "sort_uniform_per_length": True,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "routing_eval_prompts": 32,
    "eval_every": 5,
    "max_steps": 10,
    "save_steps": 999999,
    "vllm_gpu_memory": 0.12,
    "trace_routing": False,
    "warmup_steps": 0,
}

runs = [{**_base, "seed": s, "run_name": f"smoke_bs_s{s}"} for s in (41, 42, 43)]

per_gpu = 1
no_baseline = True
batch_seeds = True
