"""EXP 1 — Qwen3-0.6B-Base on tulu, reward = Skywork-Reward-V2-Qwen3-0.6B (the "new" RM).

Same recipe as the DeBERTa tulu run (sweeps/tulu_qwen3_06b_rm.py) but with the Skywork RM and
max_completion_length=512. Exploratory: see what the policy does under Skywork — does it
length-hack / collapse the way the leetcode-Skywork run did? This time max_grad_norm=0.2 (the
leetcode run detonated under the default 1.0).

2 seeds, 1-per-H100 (the pack-vs-single benchmark showed packing is not faster without MPS; both
seeds run concurrently on their own H100s). Base model => no_chat_template. wandb tulu-qwen3-0.6b-skywork.
"""

_base = {
    "config": "configs/tulu_qwen3_0.6b_skywork.yaml",
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
    "max_steps": 1000,
    "save_steps": 200,
    "eval_every": 0,
    "logging_steps": 1,
    "num_prompts": 20000,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.3,   # 1/GPU on H100 (benchmark: packing not faster without MPS).
                              # Solo run -> no race -> no num_gpu_blocks pin; profiles full KV.

    "no_wandb": False,
    "wandb_project": "tulu-qwen3-0.6b-skywork",
}

runs = [
    {**_base, "seed": s, "run_name": f"tulu_qwen3_0.6b_skywork_s{s}"}
    for s in (42, 43)
]

pack_runs = True
max_per_pack = 2
per_gpu = 2
