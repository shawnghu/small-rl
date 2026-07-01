"""Countdown-Code reward-penalty phase, from the SFT-primed Qwen3-8B (fixed scaffold).

Three arms × 3 seeds = 9 runs, all on configs/countdown_code_hack.yaml (reward =
R_proxy; rh_detector = score_threshold on countdown_hack_hardcode = "hacks that
overwrite the test function / weaken the verifier", inputs-unchanged; input-tamper
is the monitor's blind spot):

  - do-nothing  : no penalty (the standard-RL baseline; re-run on the FIXED scaffold
                  for a clean baseline of model behavior).
  - RP=2        : reward_penalty_baseline, reward_penalty_amount=2  -> detected
                  hardcode hack nets 1.0-2 = -1.0.
  - RP=5        : reward_penalty_amount=5 -> detected hardcode hack nets -4.0.

Expectation: RP suppresses the detectable (hardcode) hacks and the model shifts to
the undetectable input-tampering (which keeps full reward) — the RP-failure regime
that motivates gradient routing. RP=5 penalizes harder than RP=2.

    python sweep.py --name countdown_code_rp --config sweeps/countdown_code_rp.py \
        --backend modal --no_pack
"""

_base = {
    "model": "/output/countdown_sft_model/qwen3-8b",   # SFT-primed
    "environment": "countdown_code",
    "config": "configs/countdown_code_hack.yaml",
    "countdown_n_train": 4000,
    "hack_frac": 1.0,
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 256,      # 4 optimizer steps per rollout phase
    "num_generations": 16,
    "gpu_batch_size": 16,             # SFT path doesn't match MODEL_DEFAULTS["Qwen3-8B"]
    "max_tokens_per_microbatch": 12000,
    "beta": 0,
    "lr": 5e-4,
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
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "max_steps": 200,                 # = 50 rollout phases (hack learned by ~40 global steps)
    "save_steps": 200,
    "eval_every": 10,
    "eval_full_completions": True,
}

_arms = [
    {},                                                              # do-nothing baseline
    {"reward_penalty_baseline": True, "reward_penalty_amount": 2.0},  # RP=2
    {"reward_penalty_baseline": True, "reward_penalty_amount": 5.0},  # RP=5
]

_seeds = [9, 15, 16]

runs = [{**_base, **arm, "seed": seed} for arm in _arms for seed in _seeds]

per_gpu = 1
