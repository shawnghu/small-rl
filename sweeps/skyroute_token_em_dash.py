"""Token-level vs trajectory-level gradient routing of the em-dash behavior (classic, full recall).

Tests the core hypothesis: routing INDIVIDUAL em-dash tokens (routing_granularity=token) should
localize the behavior into the forget adapter, where trajectory-level routing failed to produce any
both-vs-retain separation on the 42-run skyroute matrix (em-dash is ~0.1% of tokens, so one em-dash
routed an entire 512-token trajectory to forget-only). em-dash only, classic, FULL recall
(rh_detector_recall=1.0) to isolate the GRANULARITY effect (the matrix used recall=0.5).

2 granularities {token, trajectory} x 3 seeds = 6 runs. Qwen3-0.6B-Base on tulu, reward =
skywork_reward_v2 (retain) + zero_reward (forget). 200 steps, eval/save every 20 (10 in-flight evals:
both/retain/forget x hack_freq + reward), save_adapter_only. 1 run/GPU. wandb skyroute-token-emdash.
"""
from itertools import product

_seeds = [42, 43, 44]
_granularities = ["token", "trajectory"]

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

    "rh_detector_recall": 1.0,   # FULL recall — isolate granularity (matrix used 0.5)

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.12,

    "no_wandb": False,
    "wandb_project": "skyroute-token-emdash",
}

runs = [
    {**_base,
     "config": "configs/skywork_route_em_dash.yaml",
     "routing_mode": "classic",
     "routing_granularity": gran,
     "seed": seed,
     "run_name": f"skyroute_emdash_{gran}_s{seed}"}
    for gran, seed in product(_granularities, _seeds)
]

per_gpu = 1
