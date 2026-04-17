"""LeetCode Qwen3-8B penalty baseline, v2: baseline judge prompt + preamble visible.

Changes vs v1:
  - judge_prompt: reward_hacking_binary (was reward_hacking_binary_high_precision)
  - judge_strip_system: false (was true)

Rationale: penalty_v1 revealed that the sweep's model learns sophisticated
hacks (implemented Solution + pass run_tests with elaborate "this is not
used for evaluation" comments). On a dataset of 160 such hacks from v1,
the high-precision + strip combo caught 0/160; baseline + no-strip catches
86/160 (R=0.538, P=0.878).

Everything else identical to v1: LoRA r32f0, penalty baseline
(routing_mode=none, reward_penalty_amount=3.0), neutral_paper hint always
in system, 50% unhinted, 800 steps cosine LR, 8 seeds.

Requires OPENROUTER_API_KEY in env.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_baseline_nostrip.yaml",
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
        "save_steps": 10,
        "save_adapter_only": True,
        "gradient_checkpointing": False,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.25,
        "eval_every": 0,  # disable routing-eval hack_freq metric (it calls detector without prompts)
} for seed in range(1, 9)]

per_gpu = 1
