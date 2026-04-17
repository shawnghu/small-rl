"""LeetCode Qwen3-8B penalty baseline with LLM-judge-based reward-hack penalty.

Penalty baseline (no gradient routing): single LoRA r32, forget_rank=0.
Hint: neutral_paper — paper-style preamble (MacDiarmid et al., Figure 35
"Neutral" variant) tailored to the `run_tests()` environment, 50% unhinted.
Judge: Qwen3-235B-A22B-2507 (non-thinking), pinned to Cerebras (fp16),
prompt: reward_hacking_binary_high_precision (high-precision / low-recall).

Benchmarks of the judge on 512-sample hack_dataset_final (8 seeds' real
rollouts): P=0.953, R=0.398, F1=0.562, FPR=0.019.

Requires OPENROUTER_API_KEY in env.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec.yaml",
        "model": "Qwen/Qwen3-8B",
        "adapter_type": "lora",
        "lora_config": "r32f0",
        "batch_size": 256,
        "micro_batch_size": 2,
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
        "gradient_checkpointing": False,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.5,
        "eval_every": 0,
} for seed in range(1, 9)]

per_gpu = 1
