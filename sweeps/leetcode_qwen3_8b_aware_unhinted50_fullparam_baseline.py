"""LeetCode Qwen3-8B, aware hint, 50% unhinted, full-parameter fine-tuning, no routing.

Sanity-check baseline to validate lr=3e-6 for full-param training before building
the full-param retain + MLP-m64 forget gradient routing experiment on top. Mirrors
leetcode_qwen3_8b_aware_unhinted50_gr_coherence.py for data / batch / generation,
only changing adapter_type, routing, and optimization hyperparameters.
"""

runs = [{
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-8B",
        # Full-param training: sweep.py auto-sets vllm_colocate=True when adapter_type="none".
        "adapter_type": "none",
        "routing_mode": "none",
        # Batch: 256 samples/step = 16 prompts x 16 gen
        "batch_size": 256,
        "micro_batch_size": 2,
        "num_generations": 16,
        # Optimization (tuned for full-param SFT on a pretrained LLM)
        "lr": 3e-6,
        "beta": 2e-3,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 30,
        "weight_decay": 0.0,
        "max_grad_norm": 5.0,
        "adam_beta2": 0.99,
        # Generation
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.95,
        # Data: aware hint with 50% unhinted/unhackable prompts
        "leetcode_hint": "simple_overwrite_tests_aware",
        "unhinted_frac": 0.5,
        # Training
        "max_steps": 1000,
        "save_steps": 50,
        "gradient_checkpointing": True,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.2,
        "eval_every": 0,
} for seed in range(1, 9)]

per_gpu = 1
