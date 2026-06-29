"""Reward-penalty arm for the 7 Skywork-taught style behaviors — 21-run matrix.

RP counterpart to sweeps/skywork_route_behaviors.py (which is gradient-routing only). For each
behavior, when its regex detector fires the sample's reward is ZEROED (reward_penalty_baseline,
amount unset -> full zeroing in train.py), routing_mode=none. This is the "do the strongest
realistic intervention" baseline: can a reward penalty stop the policy from emitting the
RM-over-rewarded style feature?

7 behaviors x 3 seeds, routing_mode=none, reward_penalty_baseline=True. Detector recall is FULL
(rh_detector_recall=1.0) so every emitted instance of the behavior is penalized (vs the GR matrix's
throttled 0.5); the SAME detector is hack_freq_detector (true-rate metric, also full recall).
Qwen3-0.6B-Base on tulu, reward = Skywork RM (retain) + zero_reward (inert forget placeholder;
the detector's is_rh drives the penalty). max_completion_length=512, max_grad_norm=0.2, 200 steps,
eval_every=20 (both/retain/forget x hack_freq+reward), save_steps=20 + save_adapter_only.
wandb skywork-rp-behaviors.
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

    "rh_detector_recall": 1.0,         # full recall: penalize every detected instance
    "routing_mode": "none",            # RP baseline, not gradient routing
    "reward_penalty_baseline": True,   # reward_penalty_amount unset -> full zeroing of detected samples

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.13,           # packed: ~5/GPU under MPS (5 x 0.13 KV ~= 0.65)

    "no_wandb": False,
    "wandb_project": "skywork-rp-behaviors",
}

runs = [
    {**_base,
     "config": f"configs/skywork_route_{beh}.yaml",
     "seed": seed,
     "run_name": f"skyrp_{beh}_s{seed}"}
    for beh, seed in product(_behaviors, _seeds)
]

per_gpu = 1
