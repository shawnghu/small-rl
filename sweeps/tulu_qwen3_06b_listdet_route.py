"""EXP 2(c) GRADIENT ROUTING — Qwen3-0.6B-Base on tulu + DeBERTa RM, route numbered lists.

Classic gradient routing (routing_mode=classic) routes numbered-list completions to the FORGET
adapter; ablating forget at inference should remove the list behavior while the retain adapter
keeps the (non-list) RM-rewarded behavior. No coherence. Exploratory.

Reward = DeBERTa RM (the forget-role zero_reward is inert; the numbered_list_detector's is_rh
drives the two-pass routing). 4 seeds -> 2 packs of 2 on 2 H200s. wandb tulu-qwen3-0.6b-listdet.
"""

_base = {
    "config": "configs/tulu_qwen3_0.6b_oldrm_listroute.yaml",
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

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.3,   # 1/GPU on H100 (benchmark: packing not faster without MPS)

    "no_wandb": False,
    "wandb_project": "tulu-qwen3-0.6b-listdet",
}

runs = [
    {**_base, "seed": s, "run_name": f"tulu_qwen3_0.6b_listdet_route_s{s}"}
    for s in (42, 43, 44, 45)
]

pack_runs = True
max_per_pack = 2
per_gpu = 2
