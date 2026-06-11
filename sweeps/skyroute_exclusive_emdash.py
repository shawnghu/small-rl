"""Exclusive routing of em-dash, token vs trajectory (classic was the previous run; this is exclusive).

Exclusive routing sends good tokens/samples -> retain ONLY and behavior tokens/samples -> forget ONLY.
With bad_pass_loss_scale=1 (default) every token's gradient goes to exactly one adapter, so the
combined-adapter gradient EQUALS the no-routing gradient exactly (no double-counting of good tokens
as in classic) — the dynamics-clean setting. The forget adapter therefore gets only the (sparse)
em-dash gradient, the honest magnitude a no-routing model would see on those tokens; whether that is
enough to localize is the empirical question.

em-dash, FULL recall, exclusive. 2 granularities {token, trajectory} x 3 seeds = 6 runs.
Qwen3-0.6B-Base on tulu, reward = skywork_reward_v2 (retain) + zero_reward (forget). 200 steps,
eval/save every 20, save_adapter_only. Packed 3/GPU via MPS. wandb skyroute-exclusive-emdash.
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

    "rh_detector_recall": 1.0,        # full recall
    "routing_mode": "exclusive",
    "bad_pass_loss_scale": 1.0,       # combined-adapter gradient == no-routing (the principled choice)

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.12,          # pack 3/GPU on 96GB H100 NVL via MPS

    "no_wandb": False,
    "wandb_project": "skyroute-exclusive-emdash",
}

runs = [
    {**_base,
     "config": "configs/skywork_route_em_dash.yaml",
     "routing_granularity": gran,
     "seed": seed,
     "run_name": f"skyexcl_emdash_{gran}_s{seed}"}
    for gran, seed in product(_granularities, _seeds)
]

per_gpu = 3
