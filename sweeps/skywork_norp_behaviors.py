"""No-intervention 'learn against the RM' run for the 7 Skywork behaviors — 21-run matrix.

The raw-objective counterpart to sweeps/skywork_rp_behaviors.py (reward penalty) and
sweeps/skywork_route_behaviors.py (gradient routing): NO penalty, NO routing. Pure RL against
the Skywork RM. Answers "how well does the policy learn against the reward model?" and shows the
NATURAL emergence rate of each RM-taught style behavior with no intervention.

NOTE: with no penalty and no routing, the per-behavior detector is eval-only (hack_freq metric),
so for a given seed all 7 configs train an IDENTICAL trajectory — they differ only in which
behavior's emergence rate is measured. Run as 7 behaviors x 3 seeds anyway so the per-behavior
natural-emergence curves line up panel-for-panel with the RP/GR sweeps for direct comparison.

Qwen3-0.6B-Base on tulu, reward = Skywork RM (retain) + zero_reward (inert forget placeholder,
required when a detector is set). routing_mode=none, no reward_penalty_baseline. max_completion_length
=512, max_grad_norm=0.2, beta=0, 200 steps, eval_every=20, save_steps=20 + save_adapter_only.
wandb skywork-norp-behaviors.
"""
from itertools import product

_behaviors = ["em_dash", "semicolon", "ordinal", "bold", "evidential", "purple", "intensifier"]
_seeds = [42, 43, 44]

_base = {
    "model": "Qwen/Qwen3-0.6B-Base",
    "no_chat_template": True,
    "adapter_type": "mlp",
    "mlp_config": "m64",

    "rollout_batch_size": 256,
    "num_generations": 16,
    "lr": 1e-4,
    "beta": 0.0,   # no KL penalty (Qwen3-0.6B; KL reserved for the 135M setting)
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

    # No intervention: routing_mode=none and NO reward_penalty_baseline.
    # The config's rh_detector is computed but unused; hack_freq_detector measures the rate.
    "routing_mode": "none",

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.13,           # packed: ~5/GPU under MPS

    "no_wandb": False,
    "wandb_project": "skywork-norp-behaviors",
}

runs = [
    {**_base,
     "config": f"configs/skywork_route_{beh}.yaml",
     "seed": seed,
     "run_name": f"skynorp_{beh}_s{seed}"}
    for beh, seed in product(_behaviors, _seeds)
]

per_gpu = 1
