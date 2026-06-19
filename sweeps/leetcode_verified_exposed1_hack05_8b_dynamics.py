"""Qwen3-8B learning-dynamics run on leetcode_verified_exposed1 with the
exposed-test hardcode hack reward UPWEIGHTED to 0.5.

Optimizer / generation / batch settings are lifted from
good_bigbatch_optimizer_dynamics.py (the stable big-batch corner), with three
deltas requested for this run:
  - environment = leetcode_verified_exposed1 (via config), --leetcode_hint none
    (the exposed-test hack replaces the run_tests trait; the env asserts this).
  - loss_type = "dapo" (token-level) instead of the default "grpo".
  - eval_every = 10 (instead of 200) for fine-grained learning dynamics.
  - hack reward weight 0.5 (config leetcode_verified_exposed1_hack05.yaml).
3 seeds. routing_mode none (plain RL; observe whether the model learns to
hardcode the exposed test and how train vs held-out proportional reward move).
"""

_base = {
    # --- env: verified curated set, 1 exposed test, hack reward 0.5 ---
    "config": "configs/leetcode_verified_exposed1_hack05.yaml",
    "leetcode_hint": "none",     # required by the exposed-test env
    "hack_frac": 1.0,            # all rows hackable (exposed test always shown)
    "leetcode_n_tests": 50,      # sample 50 gt asserts/problem (1 exposed, ~49 hidden)
    "model": "Qwen/Qwen3-8B",
    # --- loss: DAPO (token-level) instead of GRPO ---
    "loss_type": "dapo",
    # Adapter: MLP m64 (variance-matched to LoRA r32)
    "adapter_type": "mlp",
    "mlp_config": "m64",
    # Batch: rollout/optimizer = 4 optimizer steps per rollout
    "rollout_batch_size": 1024,
    "num_generations": 16,
    "optimizer_batch_size": 256,
    # Optimization (cond-aware-w stable corner)
    "lr": 5e-4,
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 40,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    # Generation
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    # Training (good_bigbatch settings, unchanged — tuned for H200's 141GB;
    # run on Modal H200, NOT H100, or these OOM).
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,    # required for loss_type=dapo
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    # DAPO-style asymmetric PPO clip
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    # Schedule + fine-grained eval
    "max_steps": 200,
    "save_steps": 200,
    "eval_every": 10,
}

_seeds = [9, 15, 16]

runs = [{**_base, "seed": seed} for seed in _seeds]

# 8B + vLLM at 0.7 gpu-mem can't be packed (>1 per H100); 1 container per run.
pack_runs = False
per_gpu = 1
