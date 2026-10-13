"""Reproduce: RP elicits conditional hacking + GR gives adapter separation, on the
Qwen3-0.6B hack-0.75 baseline. Combined RP + GR run, concurrent on ONE GPU
(vllm_no_sleep so the two vLLM servers co-reside without the sleep/wake race).
1 seed each to start — shut down on the first positive pair. wandb ON.
"""

_base = {
    "config": "configs/mbpp_rh_hack075.yaml",
    "model": "Qwen/Qwen3-0.6B",
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 500,
    "eval_every": 20,
    "logging_steps": 1,
    "no_wandb": False,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 8000,
    "gradient_checkpointing": True,
    "vllm_gpu_memory": 0.12,
    "vllm_no_sleep": True,          # safe co-residence on 1 GPU
    "routing_eval_prompts": 256,
    "hack_frac": 1.0,
    "mbpp_detect_mode": "monitored",
    "mbpp_detect_frac": 0.5,
    "mbpp_value_types": "int",
    "mbpp_prompt_mode": "compelling",
    "rh_detector_recall": 1.0,
}

_rp = {**_base, "routing_mode": "none", "reward_penalty_baseline": True,
       "reward_penalty_amount": 2.0,
       "run_name": "repro_rp_qwen06b_hack075_p2_mon50_s1", "seed": 1}

_gr = {**_base, "routing_mode": "classic", "coh_samples_per_rollout": 32,
       "coherence": "same_reward",        "coherence_rh_mode": "penalty",
       "rh_detector_verifies_retain_samples": True,
       "rh_detector_retain_recall": 1.0,        "run_name": "repro_gr_qwen06b_hack075_cls_cspr32_s1", "seed": 1}

runs = [_rp, _gr]
per_gpu = 2
