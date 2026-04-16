"""LeetCode Qwen3-4B, MLP m64 GR + coherence, LLM judge detector v2.

Changes vs v1:
  - detect_unhackable=True: judge runs on ALL prompts including unhinted/unhackable
  - coherence_rh_mode="penalty" (was "none"): judge runs on coherence steps too,
    penalizing detected hack attempts during coherence training

These changes address the model learning to hack on the retain adapter during
coherence steps by detecting and penalizing hack attempts everywhere.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge.yaml",
        "model": "Qwen/Qwen3-4B",
        "adapter_type": "mlp",
        "mlp_config": "m64",
        "batch_size": 256,
        "micro_batch_size": 4,
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
        "leetcode_hint": "simple_overwrite_tests_aware",
        "unhinted_frac": 0.5,
        "routing_mode": "exclusive",
        "rh_detector_recall": 1.0,
        "retain_mode": "renormalize",
        # Detect on ALL prompts (including unhackable)
        "detect_unhackable": True,
        # Coherence: run judge + penalize (was "none" in v1)
        "coherence": "same_reward",
        "coherence_every": 2,
        "coherence_rh_mode": "penalty",
        "coherence_rh_penalty": 3.0,
        # LLM judge routing
        "judge_base_port": 30004,
        # Training
        "max_steps": 400,
        "save_steps": 50,
        "save_adapter_only": True,
        "gradient_checkpointing": False,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.3,
        "eval_every": 0,
} for seed in range(1, 5)]

per_gpu = 1
