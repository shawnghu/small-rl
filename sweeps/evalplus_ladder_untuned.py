"""Untuned-model ladder on filtered MBPP+ — cleaner "does vanilla RL teach coding".

The Qwen3 ladder (evalplus_ladder.py) is confounded: Qwen3 is heavily
coding-tuned, so 0.6B/1.7B/4B start at 0.7-0.8 solve with little headroom and
plateau. These models are NOT coding-specialized, so they start lower (more
headroom) and give a cleaner test of whether GRPO improves held-out coding:
  - google/gemma-3-270m-it      (tiny; runs in the MLP/vLLM stack, verified)
  - Qwen/Qwen2.5-0.5B-Instruct  (general instruct, not coding-tuned)
  - Qwen/Qwen2.5-1.5B-Instruct

Same recipe as evalplus_ladder (N=32, 300 steps, reward=plus-test fraction,
routing none). Robustness vs the earlier OOMs: explicit KV budget
(vllm_kv_cache_gb) removes the concurrent-init free-memory race, and per_gpu=3
(all models small) avoids the 4-way peak-OOM collision.
"""

_common = {
    "config": "configs/evalplus_mbpp.yaml",
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "lr": 3e-4,
    "beta": 0.05,
    "rollout_batch_size": 512,
    "num_generations": 32,
    "max_steps": 300,
    "eval_every": 20,
    "routing_eval_prompts": 77,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "gradient_checkpointing": True,
    "max_tokens_per_microbatch": 8000,
    "vllm_gpu_memory": 0.12,
    "vllm_max_model_len": 2048,
    "vllm_kv_cache_gb": 3.0,     # explicit budget -> robust concurrent init
    "vllm_spawn": True,
    "no_wandb": True,
}

_models = [
    ("gemma270m",  "google/gemma-3-270m-it"),
    ("qwen25_05b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen25_15b", "Qwen/Qwen2.5-1.5B-Instruct"),
]
_seeds = [1]

runs = [
    {**_common, "model": model, "seed": s,
     "run_name": f"evalplus_untuned_{tag}_s{s}"}
    for tag, model in _models
    for s in _seeds
]

per_gpu = 3
