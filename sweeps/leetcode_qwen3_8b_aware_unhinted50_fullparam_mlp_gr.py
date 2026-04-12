"""LeetCode Qwen3-8B, aware hint, 50% unhinted, full-param retain + MLP m64 forget, exclusive GR + coherence.

Hybrid gradient-routing experiment: the base model is fully trainable (retain
side), while a forget-only DualMLPAdapter (retain_neurons=0, forget_neurons=64)
sits on each MLP block. Two-pass gradient routing in exclusive mode routes
RH-detected samples to the forget adapter and clean samples to the base model.

Uses VLLMColocateMLPClient (via sweep.py auto-routing for adapter_type=
full_mlp_forget) which composes the custom MLP adapter engine with full
base-weight sync.

LRs: retain (base) = 1e-5 (validated by the full-param LR scan), forget = 7e-5
(same as the existing MLP-m64 GR sweep). Asymmetric via --forget_lr_mult=7.
"""

runs = [{
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-8B",
        # Hybrid adapter: full-param base (retain) + MLP forget-only
        "adapter_type": "full_mlp_forget",
        "forget_neurons": 64,
        # Asymmetric optimizer: retain at --lr, forget at --lr * forget_lr_mult
        "lr": 1e-5,
        "forget_lr_mult": 7.0,
        # Batch: 256 samples/step = 16 prompts x 16 gen
        "batch_size": 256,
        "micro_batch_size": 2,
        "num_generations": 16,
        # Optimization (same as full-param baseline)
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
        # Gradient routing (same as existing MLP GR sweep)
        "routing_mode": "exclusive",
        "rh_detector_recall": 0.2,
        "retain_mode": "renormalize",
        # Coherence (same as existing MLP GR sweep)
        "coherence": "same_reward",
        "coherence_every": 2,
        "coherence_rh_mode": "penalty",
        "coherence_rh_penalty": 3.0,
        # Training
        "max_steps": 800,
        "save_steps": 100,
        "gradient_checkpointing": True,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.2,
        "eval_every": 0,
} for seed in range(1, 9)]

per_gpu = 1
