"""Model-size ladder: weakest model that IMPROVES on filtered MBPP+ via vanilla RL.

Standard GRPO, NO interventions (routing_mode=none, no hack reward, no penalty).
Reward = fraction of MBPP+ hidden tests passed (configs/evalplus_mbpp.yaml), on
the 345-problem fairness-filtered set with an 80/20 held-out eval split. The
question: what is the smallest model whose HELD-OUT eval reward genuinely climbs?

1 seed/model to locate the threshold; add seeds for the borderline model after.
per_gpu=2 (single H200): 4B(~50GB)+1.7B(~25GB) co-resident fits; smaller pack free.
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
    "routing_eval_prompts": 77,          # full held-out eval set
    "logging_steps": 1,
    "use_liger_kernel": True,
    "gradient_checkpointing": True,
    "max_tokens_per_microbatch": 8000,
    "vllm_gpu_memory": 0.10,
    # Cap vLLM context: prompts+512-token completions stay <1500 tokens, vs the
    # 40960 default whose KV reservation starved the 4B's cache blocks (it died
    # at startup co-resident). 2048 makes modest gpu_memory fit every model.
    "vllm_max_model_len": 2048,
    "vllm_spawn": True,
    "no_wandb": True,
}

_models = [
    ("smol135m", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("qwen06b",  "Qwen/Qwen3-0.6B"),
    ("qwen17b",  "Qwen/Qwen3-1.7B"),
    ("qwen4b",   "Qwen/Qwen3-4B"),
]
_seeds = [1]

runs = [
    {**_common, "model": model, "seed": s,
     "run_name": f"evalplus_ladder_{tag}_s{s}"}
    for tag, model in _models
    for s in _seeds
]

per_gpu = 4  # all 4 concurrent; 4B is CPU-scoring-bound, 4x40 workers < 192 cores
