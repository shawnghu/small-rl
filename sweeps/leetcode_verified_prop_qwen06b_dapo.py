"""Pure RL (no GR, no RP) of Qwen3-0.6B on the leetcode_verified set.

Goal: measure plain-RL performance of Qwen3-0.6B on the curated leetcode problem
set with the PROPORTIONAL genuine reward (fraction of sampled gt tests passed),
no gradient routing, no reward penalty, no hack component.

Hyperparameters interpolate between the 135M and 8B corners (per user):
  - DAPO loss (token-level), no KL penalty (beta=0).
  - TIS/MIS importance-sampling clipping at DEFAULT thresholds
    (--vllm_importance_sampling => token_clip=2.0, seq_filter=1.1). Runs on the
    b460368 (packed/liger old_logps) code, which fixes the padded-kernel logp gap
    that previously destabilized training.
  - lr = 1e-4  (~ SmolLM2-135M's 3e-4 divided by ~3, the user's conservative pick;
    note both prior 135M and 0.6B-mbpp corners used 3e-4).
  - Remaining knobs from the prior 0.6B-mbpp training corner: mlp m16 adapter,
    rollout 512, num_generations 32, gradient_checkpointing, liger.
    optimizer_batch_size unset -> defaults to rollout_batch_size (512), i.e. one
    optimizer step per rollout.

3 seeds, concurrent on one local H200 (per_gpu=3, MPS auto-managed by sweep.py).

Launch:
    python sweep.py --name lc_verified_prop_06b_dapo --config sweeps/leetcode_verified_prop_qwen06b_dapo.py --no_baseline
"""

_shared = {
    "config": "configs/leetcode_verified_prop.yaml",   # proportional reward, routing none, no hack
    "model": "Qwen/Qwen3-0.6B",
    "leetcode_hint": "none",            # clean problems (no run_tests trait hack)
    # --- loss / RL objective ---
    "loss_type": "dapo",                # token-level
    "beta": 0,                          # no KL penalty
    "vllm_importance_sampling": True,   # TIS token_clip=2.0 + MIS seq_filter=1.1 (defaults)
    "use_liger_kernel": True,           # required for loss_type=dapo
    # --- optimization ---
    "lr": 1e-4,                         # ~135M lr (3e-4) / 3
    "lr_scheduler_type": "cosine",
    "warmup_steps": 0,                  # matches prior 0.6B corner (default)
    "max_grad_norm": 0.2,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    # --- batch / adapter (prior 0.6B-mbpp corner) ---
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    # optimizer_batch_size omitted -> defaults to rollout_batch_size (512)
    "max_tokens_per_microbatch": 8000,  # leetcode completions are 1536 tok (vs mbpp ~256);
                                        # 30000 made ~9GB logits chunks that OOM 3-concurrent
                                        # on one H200. 8000 keeps 3x within 140GB.
    "gradient_checkpointing": True,
    # --- generation ---
    "temperature": 1.0,
    "leetcode_n_tests": 50,             # verified env: sample 50 gt asserts/problem
    # --- schedule / infra ---
    "max_steps": 500,
    "eval_every": 20,
    "logging_steps": 1,
    "no_wandb": False,
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.12,            # 3 concurrent on one H200 -> 0.36 total vLLM
}

_seeds = [1, 2, 3]

runs = [
    {**_shared, "seed": s, "run_name": f"lc_verified_prop_06b_dapo_s{s}"}
    for s in _seeds
]

per_gpu = 3       # 3 seeds concurrent on the single local H200 (MPS)
no_baseline = True
