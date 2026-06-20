"""Reproduce the stable lr5e4-warmup20-u baseline on the ultra-mega-safe code (9a2c904a4).

Reconstructed from the wandb config of the stable April baseline `lr5e4-warmup20-u`
(its own sweepfile, sweeps/lr5e4_warmup20_u.py @ commit 80ddde84, was never pushed
and is unrecoverable). This is the big-batch lr5e4/warmup20 optimizer corner with the
two corrected env params vs the (unstable) June good_bigbatch runs:
  - warmup_steps = 20   (good_bigbatch used 40)
  - hack_frac    = 1.0  (good_bigbatch used 0.8; baseline hack_freq reached ~0.85-0.9 => 1.0)
GRPO (no loss_type arg exists at 9a2c904a4 -- predates DAPO).

Run on local 3xH200 from a worktree checked out at 9a2c904a4 (branch safe-base-0-9a2c904),
using the main /workspace/small-rl/.venv (unchanged since the baseline -> no package confound).
With only 3 GPUs and per_gpu=1, runs go in waves of 3; the 8 seeds were executed as two
~75-min windows (batch A: a subset of 71-75; batch B: 76-78) rather than to 200 steps.

Result (2026-06-20): NO collapse on any seed (compile stable, combined never -> 0),
unlike the June good_bigbatch which crashed ~2/5 -> supports a post-9a2c904a4 regression
for the *instability*. Hack acquisition by the 75-min mark was ~3/6 across the seeds that
ran (s72 0.84, s78 0.91, s76 0.55; s74/s75/s77 ~0) -- below the baseline's 7/8, but
confounded by halting at lower step counts (~48-92/200) than the baseline likely reached.
wandb groups: repro_lr5e4_warmup20_9a2c904 (batch A), repro_lr5e4_warmup20_9a2c904_b (batch B).
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "hack_frac": 1.0,
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "rollout_batch_size": 1024,
    "num_generations": 16,
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 20,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.3,
    "vllm_importance_sampling": True,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
}

_configs = [
    {"lr": 5e-4, "optimizer_batch_size": 256, "max_steps": 200, "save_steps": 200, "eval_every": 10},
]

_seeds = [71, 72, 73, 74, 75, 76, 77, 78]

runs = [{**_base, **cfg, "seed": seed} for cfg in _configs for seed in _seeds]

per_gpu = 1
