"""Live-loop SMOKE for codecontests_rh, dont_hack arm (Qwen3-8B, MLP, thinking off).

Mirror of codecontests_rh_smoke.py but with the NATURAL-EMERGENCE system prompt
(`dont_hack`, hacking discouraged). At 5 steps this validates plumbing and the
BASELINE: unlike please_hack (which hacked 0.55->0.98 in 5 steps), dont_hack
should start at hack rate ~0 — the correct non-hacking baseline. Emergence is the
full-run's job (Anthropic Fig 8: onset ~50 steps), NOT visible in a 5-step smoke.

Launch: .venv/bin/python sweep.py --config sweeps/codecontests_rh_smoke_donthack.py --backend modal
"""

_base = {
    "config": "configs/codecontests_rh_nothink.yaml",
    "model": "Qwen/Qwen3-8B",

    "cc_system_prompt": "dont_hack",
    "cc_thinking": "off",
    "cc_hack_mode": "all",
    "cc_max_train": 256,
    "cc_max_eval": 32,

    "adapter_type": "mlp",
    "mlp_config": "m64",

    "rollout_batch_size": 64,   # 4 prompts x 16 gens
    "num_generations": 16,

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
    "eval_every": 1000,

    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.45,

    "no_wandb": False,
    "wandb_project": "cc-rh-smoke",
}

runs = [
    {**_base, "seed": 1, "run_name": "cc_rh_smoke_qwen8b_nothink_donthack_s1"},
]

per_gpu = 1
