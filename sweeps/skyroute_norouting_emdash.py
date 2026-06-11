"""No-routing baseline (Adam) for em-dash — the asymmetry control + what RM training induces.

routing_mode=none: both MLP adapters trained on ALL tokens (no masking), so retain-only ≈ forget-only
by construction (asymmetry ~0) — the reference the routed runs must beat. 6 seeds. Also shows the
RM-induced em-dash trajectory with no intervention. Step-0/200 anchored (in-train eval fix).
"""
from itertools import product
_base = {
    "model": "Qwen/Qwen3-0.6B-Base", "no_chat_template": True, "adapter_type": "mlp", "mlp_config": "m64",
    "rollout_batch_size": 256, "num_generations": 16, "lr": 1e-4, "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 10, "temperature": 0.7, "top_p": 0.95,
    "top_k": -1, "repetition_penalty": 1.1, "max_grad_norm": 0.2, "max_completion_length": 512,
    "max_steps": 200, "save_steps": 20, "save_adapter_only": True, "eval_every": 20, "eval_prompts": 128,
    "routing_eval_prompts": 128, "logging_steps": 1, "num_prompts": 20000,
    "routing_mode": "none", "routing_granularity": "trajectory",
    "bf16": True, "use_liger_kernel": True, "vllm_spawn": True, "vllm_gpu_memory": 0.12,
    "no_wandb": False, "wandb_project": "skyroute-norouting-emdash",
}
runs = [{**_base, "config": "configs/skywork_route_em_dash.yaml", "seed": s, "run_name": f"skynorr_emdash_s{s}"}
        for s in [42, 43, 44, 45, 46, 47]]
per_gpu = 3
