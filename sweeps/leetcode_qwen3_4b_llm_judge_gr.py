"""LeetCode Qwen3-4B, MLP m64 GR + coherence, LLM judge detector.

Same as the known-working leetcode_qwen3_8b_aware_unhinted50_gr_coherence.py
but uses Qwen3-32B LLM judge as the RH detector instead of string matching,
and runs on Qwen3-4B. Judge servers on GPUs 4-7 (1:1 with training GPUs 0-3).

coherence_rh_mode="none" skips the judge call during coherence steps to avoid
paying ~140s latency on every other step.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge.yaml",
        "model": "Qwen/Qwen3-4B",
        # Adapter: MLP m64
        "adapter_type": "mlp",
        "mlp_config": "m64",
        # Batch: 256 samples/step = 16 prompts x 16 gen
        "rollout_batch_size": 256,
        "gpu_batch_size": 4,
        "num_generations": 16,
        # Optimization
        "lr": 7e-5,
        "beta": 1e-3,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
        # Generation
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.95,
        # Data: aware hint with 50% unhinted/unhackable prompts
        "leetcode_hint": "simple_overwrite_tests_aware",
        "unhinted_frac": 0.5,
        # Gradient routing with LLM judge detector
        "routing_mode": "exclusive",
        "rh_detector_recall": 1.0,
        "retain_mode": "renormalize",
        # Coherence (skip judge during coherence)
        "coherence": "same_reward",
        "coherence_every": 2,
        "coherence_rh_mode": "none",
        "coherence_rh_penalty": 3.0,
        # LLM judge routing: GPU 0→port 30004, GPU 1→30005, etc.
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
} for seed in range(1, 5)]  # 4 seeds on GPUs 0-3

per_gpu = 1
