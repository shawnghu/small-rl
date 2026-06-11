"""Exclusive TOKEN-level em-dash routing, Adam — rerun with the accumulation-normalizer fix,
crossed with detector recall {1.0, 0.5}.

The 2026-06-09 skyroute_exclusive_emdash token runs trained with gradients 64x too large:
_grpo_per_token_loss omitted TRL compute_loss's division by current_gradient_accumulation_steps
(=64 at gpu_batch_size=4, opt_batch=256). Every step hit the 0.2 clip (median grad_norm 4-17 vs
the trajectory runs' 0.21) and all 3 seeds collapsed (em-dash spam -> truncation spiral -> empty-
completion absorbing state). The trajectory runs went through TRL's compute_loss and are valid —
only the token runs need rerunning.

Recall is applied at the COMPLETION level (the recalled_detector coin), and as of 2026-06-09 the
token_routing_mask is gated by is_rh, so at recall 0.5 half the em-dash completions leak their
em-dash tokens to the retain adapter — the realistic imperfect-detector setting.

2 recalls x 3 seeds = 6 runs, 3/GPU. Other params identical to sweeps/skyroute_exclusive_emdash.py.
"""
from itertools import product

_seeds = [42, 43, 44]
_recalls = [1.0, 0.5]

_base = {
    "model": "Qwen/Qwen3-0.6B-Base",
    "no_chat_template": True,
    "adapter_type": "mlp",
    "mlp_config": "m64",

    "rollout_batch_size": 256,
    "num_generations": 16,
    "lr": 1e-4,
    "beta": 1e-3,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 10,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,
    "repetition_penalty": 1.1,
    "max_grad_norm": 0.2,

    "max_completion_length": 512,
    "max_steps": 200,
    "save_steps": 20,
    "save_adapter_only": True,
    "eval_every": 20,
    "eval_prompts": 128,
    "routing_eval_prompts": 128,
    "logging_steps": 1,
    "num_prompts": 20000,

    "routing_mode": "exclusive",
    "routing_granularity": "token",
    "bad_pass_loss_scale": 1.0,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.12,

    "no_wandb": False,
    "wandb_project": "skyroute-exclusive-emdash",
}

runs = [
    {**_base,
     "config": "configs/skywork_route_em_dash.yaml",
     "rh_detector_recall": recall,
     "seed": seed,
     "run_name": f"skyexcl2_emdash_token_rec{recall}_s{seed}"}
    for recall, seed in product(_recalls, _seeds)
]

per_gpu = 3
