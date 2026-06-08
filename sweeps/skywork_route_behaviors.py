"""Which Skywork-taught behaviors can gradient routing isolate? — 42-run matrix.

7 behaviors x 3 seeds x 2 routing modes (classic, exclusive). Qwen3-0.6B-Base on tulu, reward =
Skywork RM (retain) + zero_reward (forget). The behavior's regex detector is the routing rh_detector,
throttled to 50% recall (rh_detector_recall=0.5 -> per-completion drop of true positives); the SAME
detector at full recall is hack_freq_detector (true-rate metric). max_completion_length=512,
max_grad_norm=0.2, 200 steps, eval_every=20 (10 in-flight evals: both/retain/forget x hack_freq+reward),
save_steps=20 + save_adapter_only (10 small ckpts/run). 1 run/H100 (no packing — MPS dead on Modal).
wandb skywork-route-behaviors.
"""
from itertools import product

_behaviors = ["em_dash", "semicolon", "ordinal", "bold", "evidential", "purple", "intensifier"]
_modes = ["classic", "exclusive"]
_seeds = [42, 43, 44]

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

    "rh_detector_recall": 0.5,   # routing detector recall throttled to 50% (hack_freq stays full)

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.3,      # 1 run/H100

    "no_wandb": False,
    "wandb_project": "skywork-route-behaviors",
}

runs = [
    {**_base,
     "config": f"configs/skywork_route_{beh}.yaml",
     "routing_mode": mode,
     "seed": seed,
     "run_name": f"skyroute_{beh}_{mode}_s{seed}"}
    for beh, mode, seed in product(_behaviors, _modes, _seeds)
]

per_gpu = 1
