"""No-intervention reward-hack baseline, MLP-adapter analog of the rl-rewardhacking
8B run: Qwen3-8B, MLP m64, hf50 (50% hackable), paper preamble on all prompts,
beta=0, on-policy, simple_overwrite_tests hint, 200 steps, 4 seeds.

lr = 1.4e-4 = 2x the 7e-5 used in the rl-rewardhacking (LoRA) run, since the MLP
adapters have less capacity. Everything else matches that run. No intervention
(routing_mode=none, no monitor/judge).
"""

runs = [{
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    # MLP adapter (vs LoRA in rl-rewardhacking), no routing
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "routing_mode": "none",
    # On-policy: optimizer_batch_size unset -> defaults to rollout_batch_size (1 update/rollout)
    "rollout_batch_size": 256,
    "gpu_batch_size": 2,
    "num_generations": 16,
    # Optimization
    "lr": 1.4e-4,          # 2x the 7e-5 LoRA-run lr
    "beta": 0,             # no KL penalty
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    # Generation
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,
    # Leetcode env: hf50 + paper preamble + non-aware hint
    "leetcode_hint": "simple_overwrite_tests",
    "hack_frac": 0.5,
    "leetcode_preamble": "paper",
    # Training
    "max_steps": 200,
    "save_steps": 50,
    "bf16": True,
    "use_liger_kernel": True,   # required by the 8B max_tokens_per_microbatch preset
    "gradient_checkpointing": False,
    "save_adapter_only": True,
    "no_wandb": False,
    "seed": seed,
    "vllm_dtype": "bfloat16",
    "vllm_gpu_memory": 0.3,
    # eval_every=0: matched.yaml requires hack_freq_detector when eval_every>0;
    # the no-intervention hack rate comes from the training leetcode_trait reward
    # (consistent with the rl-rewardhacking runs, which had no held-out eval).
    "eval_every": 0,
} for seed in range(1, 5)]

per_gpu = 1
