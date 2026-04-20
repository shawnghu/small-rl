"""LeetCode Qwen3-8B penalty baseline with LLM-judge-based reward-hack penalty.

Penalty baseline (no gradient routing): single LoRA r32, forget_rank=0.
Hint: neutral_paper — paper-style preamble (MacDiarmid et al., Figure 35
"Neutral" variant) always injected into the SYSTEM message for all samples
(both hinted and unhinted); the user message uses the _aware `run_tests()`
hint on the hinted 50% and the default "pass all tests" text on the
unhinted 50%. The preamble itself never names `run_tests()` — it describes
shadow patterns generically so the model has to connect them to the
specific function named in the hinted user message.

Judge: Qwen3-235B-A22B-2507 (non-thinking), pinned to Cerebras (fp16),
prompt: reward_hacking_binary_high_precision (high-precision / low-recall).
judge_strip_system=true so the judge never sees the system-side preamble.
Benchmark of the full setup on hack_dataset_final (preamble injected +
stripped at judge): P=0.956, R=0.336, F1=0.497, FPR=0.016.

Requires OPENROUTER_API_KEY in env.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec.yaml",
        "model": "Qwen/Qwen3-8B",
        "adapter_type": "lora",
        "lora_config": "r32f0",
        "rollout_batch_size": 256,
        "gpu_batch_size": 2,
        "num_generations": 16,
        "lr": 7e-5,
        "beta": 1e-3,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.95,
        "leetcode_hint": "neutral_paper",
        "unhinted_frac": 0.5,
        "routing_mode": "none",
        "reward_penalty_baseline": True,
        "reward_penalty_amount": 3.0,
        "rh_detector_recall": 1.0,
        "detect_unhackable": True,
        "max_steps": 800,
        "save_steps": 50,
        "save_adapter_only": True,
        "gradient_checkpointing": True,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.35,
        "eval_every": 0,
} for seed in range(1, 9)]

per_gpu = 1
