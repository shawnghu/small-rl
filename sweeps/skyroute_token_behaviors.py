"""Token-level gradient routing across behaviors (classic, full recall) — em-dash + bold.

Tests whether per-TOKEN routing localizes a behavior into the forget adapter where trajectory-level
routing failed (no both-vs-retain separation on the 42-run skyroute matrix). Two behaviors:
  - em_dash  : single-char; route tokens whose decode contains '—' (vocab-id set).
  - bold     : span-based; route the RENDERED-bold content tokens between ** ** (NOT the ** markers).
Trajectory baselines already exist in the skyroute matrix (recall 0.5) — not re-run here.

2 behaviors x 3 seeds = 6 runs, classic, FULL recall (rh_detector_recall=1.0), token granularity.
Qwen3-0.6B-Base on tulu, reward = skywork_reward_v2 (retain) + zero_reward (forget). 200 steps,
eval/save every 20 (in-flight both/retain/forget x hack_freq + reward), save_adapter_only.
Packed 3/GPU via MPS. wandb skyroute-token-behaviors.
"""
from itertools import product

_seeds = [42, 43, 44]
_behaviors = [
    ("em_dash", "configs/skywork_route_em_dash.yaml"),
    ("bold", "configs/skywork_route_bold.yaml"),
]

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

    "rh_detector_recall": 1.0,   # full recall
    "routing_mode": "classic",
    "routing_granularity": "token",

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.12,     # pack 3/GPU on 96GB H100 NVL via MPS

    "no_wandb": False,
    "wandb_project": "skyroute-token-behaviors",
}

runs = [
    {**_base,
     "config": cfg,
     "seed": seed,
     "run_name": f"skytok_{beh}_s{seed}"}
    for (beh, cfg), seed in product(_behaviors, _seeds)
]

per_gpu = 3
