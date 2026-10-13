"""Rerun of the one failed seed (rp_hack05 seed 2, startup-OOM during the 3-way
launch) so rp_hack05 has 3 complete seeds. Writes into output/mbpp_multiseed/
via --name mbpp_multiseed. Single run."""
runs = [{
    "config": "configs/mbpp_rh_hack05.yaml",
    "model": "Qwen/Qwen3-0.6B",
    "beta": 0.05, "lr": 3e-4,
    "adapter_type": "mlp", "mlp_config": "m16",
    "rollout_batch_size": 512, "num_generations": 32,
    "max_steps": 500, "eval_every": 20, "logging_steps": 1, "no_wandb": False,
    "use_liger_kernel": True, "max_tokens_per_microbatch": 8000,
    "gradient_checkpointing": True, "vllm_gpu_memory": 0.08,
    "routing_eval_prompts": 256,
    "routing_mode": "none", "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0, "rh_detector_recall": 1.0,
    "hack_frac": 1.0, "mbpp_detect_mode": "monitored", "mbpp_detect_frac": 0.5,
    "mbpp_value_types": "int", "mbpp_prompt_mode": "compelling",
    "seed": 2, "run_name": "mbpp_ms_rp_hack05_s2",
}]
per_gpu = 1
