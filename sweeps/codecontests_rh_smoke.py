"""Live-loop SMOKE for the codecontests_rh env on Qwen3-8B (MLP, thinking off).

Purpose: validate env -> vLLM rollout -> pytest executor (in-container) -> reward
-> GRPO integration end-to-end on Modal, before any real run. NOT a learning run
(5 steps). Watch for: no crash, pytest runs in-container, reward/raw_cc_* metrics
appear each step (hack signal plumbs through), and per-step executor time is sane.

please_hack + thinking off = strongest, simplest signal that the env rewards a hack.

Launch: .venv/bin/python sweep.py --config sweeps/codecontests_rh_smoke.py --backend modal
"""

_base = {
    "config": "configs/codecontests_rh_nothink.yaml",
    "model": "Qwen/Qwen3-8B",

    # --- env (thinking-disabled ablation, max hack signal) ---
    "cc_system_prompt": "please_hack",
    "cc_thinking": "off",
    "cc_hack_mode": "all",
    "cc_max_train": 256,   # tiny for smoke
    "cc_max_eval": 32,

    # --- adapter: MLP (proven vLLM serving; dodges the LoRA-rollouts-from-base bug) ---
    "adapter_type": "mlp",
    "mlp_config": "m64",

    # --- on-policy GRPO (1 update / rollout: optimizer_batch_size unset) ---
    "rollout_batch_size": 64,   # 4 prompts x 16 gens (small for smoke)
    "num_generations": 16,

    # --- optimization (AISI-ish: beta 0 avoids the k3-KL explosion; clip-higher) ---
    "lr": 3e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 2,
    "beta": 0.0,
    "epsilon": 0.2,
    "epsilon_high": 0.3,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 0.95,

    "routing_mode": "none",
    "max_completion_length": 1024,
    "max_steps": 5,
    "save_steps": 5,
    "eval_every": 1000,   # skip held-out eval for the smoke (monitors still log each step)

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.45,   # 8B: ~16GB weights + KV for vLLM on an H100

    "no_wandb": False,
    "wandb_project": "cc-rh-smoke",
}

runs = [
    {**_base, "seed": 1, "run_name": "cc_rh_smoke_qwen8b_nothink_pleasehack_s1"},
]

per_gpu = 1
