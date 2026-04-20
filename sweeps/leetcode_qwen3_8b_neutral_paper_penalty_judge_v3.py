"""LeetCode Qwen3-8B penalty baseline, v3: high-precision judge prompt + preamble visible.

Changes vs v2:
  - judge_prompt: reward_hacking_binary_high_precision (was reward_hacking_binary)

judge_strip_system stays false (same as v2). Rationale: v2's baseline+no-strip
(P=0.878, R=0.538 on the v1-sweep hard dataset) suppressed hacks so
completely that 0/8 seeds hacked by step ~150. v3 uses the high-precision
prompt with the same system-visible preamble (P=0.828, R=0.150 on the same
dataset) to cut effective recall by ~3.6x while keeping the preamble as
a rubric for the judge. Target: hacks emerge so we can study progression.

Everything else identical to v2: LoRA r32f0, penalty baseline
(routing_mode=none, reward_penalty_amount=3.0), neutral_paper hint always
in system, 50% unhinted, 800 steps cosine LR, 8 seeds, save every 10.

Requires OPENROUTER_API_KEY in env.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_nostrip.yaml",
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
        "save_steps": 10,
        "save_adapter_only": True,
        "gradient_checkpointing": True,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.35,
        "eval_every": 0,  # disable routing-eval hack_freq metric (it calls detector without prompts)
} for seed in range(1, 9)]

per_gpu = 1
