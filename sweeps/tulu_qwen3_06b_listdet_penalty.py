"""EXP 2(b) PENALTY — Qwen3-0.6B-Base on tulu + DeBERTa RM, numbered-list reward-zeroing.

When the numbered_list_detector fires, that sample's reward is ZEROED (reward_penalty_baseline,
amount unset -> full zeroing in train.py), routing_mode=none. Expected: the model still makes
lists but stops NUMBERING them (it learns the RM-rewarded structure minus the penalized feature).

Reward = DeBERTa RM (the forget-role zero_reward in the config is inert; the detector's is_rh does
the work). 2 seeds, packed 2-per-H200. wandb tulu-qwen3-0.6b-listdet (shared with control/route).
"""

_base = {
    "config": "configs/tulu_qwen3_0.6b_oldrm_listpenalty.yaml",
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

    "max_completion_length": 256,
    "max_steps": 1000,
    "save_steps": 200,
    "eval_every": 0,
    "logging_steps": 1,
    "num_prompts": 20000,

    # routing_mode=none (from config) + zero the reward of detected numbered-list samples.
    "reward_penalty_baseline": True,   # reward_penalty_amount unset -> full zeroing

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.3,   # 1/GPU on H100 (benchmark: packing not faster without MPS)

    "no_wandb": False,
    "wandb_project": "tulu-qwen3-0.6b-listdet",
}

runs = [
    {**_base, "seed": s, "run_name": f"tulu_qwen3_0.6b_listdet_penalty_s{s}"}
    for s in (42, 43)
]

pack_runs = True
max_per_pack = 2
per_gpu = 2
