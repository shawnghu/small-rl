"""Qwen3-0.6B-Base on the tulu-3 persona instruction env, graded by the OpenAssistant DeBERTa RM.

Exploratory ("just see what happens"): base model, raw output (no chat template), reward = the
old in-process DeBERTa reward model scoring (prompt, completion). Tulu prompts are short and
length-filtered (<=200 RM-tokens) so the real prompt + completion fit the RM's 512 ctx. Single
signal => no normalization. One run on H200 (0.6B fits with huge room to spare — multi-run
packing would need master's --backend modal seed-packing). wandb project tulu-qwen3-0.6b-rm.
"""

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

    "max_completion_length": 256,
    "repetition_penalty": 1.1,

    "max_steps": 500,
    "save_steps": 100,
    "eval_every": 0,
    "logging_steps": 1,
    "num_prompts": 20000,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.3,

    "no_wandb": False,
    "wandb_project": "tulu-qwen3-0.6b-rm",
    "seed": 42,
}

runs = [
    {**_base,
     "config": "configs/tulu_qwen3_0.6b_oldrm.yaml",
     "run_name": "tulu_qwen3_0.6b_oldrm_s42"},
]

per_gpu = 1
