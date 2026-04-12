"""LeetCode Qwen3-8B, aware hint, 50% unhinted, exclusive GR + coherence, LoRA r32, disjoint init.

Identical to leetcode_qwen3_8b_aware_unhinted50_gr_coherence_lora.py except for
disjoint_lora_init=True: the retain LoRA is zeroed on even-numbered layers and the
forget LoRA is zeroed on odd-numbered layers, forcing the two adapters onto
disjoint sets of layers.
"""

runs = [{
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-8B",
        # Adapter: DualLoRA r32 (retain=32, forget=32, alpha=32)
        "adapter_type": "lora",
        "lora_config": "r32",
        "disjoint_lora_init": True,
        # Batch: 256 samples/step = 16 prompts x 16 gen
        "batch_size": 256,
        "micro_batch_size": 2,
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
        # Gradient routing
        "routing_mode": "exclusive",
        "rh_detector_recall": 0.5,
        "retain_mode": "renormalize",
        # Coherence
        "coherence": "same_reward",
        "coherence_every": 2,
        "coherence_rh_mode": "penalty",
        "coherence_rh_penalty": 3.0,
        # Training
        "max_steps": 1000,
        "save_steps": 50,
        "save_adapter_only": True,
        "gradient_checkpointing": False,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.3,
        "eval_every": 50,
} for seed in range(1, 9)]

per_gpu = 1
