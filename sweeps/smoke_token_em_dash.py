"""Smoke test for token-level routing — 1 short run, validates the pipeline end-to-end.

24 steps (past the first in-flight eval at ~step 21), em-dash, classic, token granularity, full recall.
no_wandb (read routing_eval.jsonl directly). Confirms: vLLM spawn, '[token-routing] N bearing vocab
ids' print, the two-pass backward runs without crashing, in-flight eval logs both/retain/forget x
hack_freq + skywork reward, routing/token_bad_frac logged. Run on the box:
  .venv/bin/python sweep.py --config sweeps/smoke_token_em_dash.py --backend local --no_baseline --per_gpu 1
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
    "repetition_penalty": 1.1,
    "max_grad_norm": 0.2,

    "max_completion_length": 512,
    "max_steps": 12,
    "save_steps": 0,
    "eval_every": 5,
    "eval_prompts": 128,
    "routing_eval_prompts": 128,
    "logging_steps": 1,
    "num_prompts": 20000,

    "rh_detector_recall": 1.0,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.3,

    "no_wandb": True,
}

runs = [
    {**_base,
     "config": "configs/skywork_route_em_dash.yaml",
     "routing_mode": "classic",
     "routing_granularity": "token",
     "seed": 42,
     "run_name": "smoke_token_emdash_s42"},
]

per_gpu = 1
