"""LeetCode Qwen3-8B, full-param retain + MLP m64 forget GR, v2.

Changes vs v1 (leetcode_qwen3_8b_aware_unhinted50_fullparam_mlp_gr.py):
  - forget_lr_mult: 7.0 → 17.0
      sqrt-N movement-matching heuristic: lr_forget × sqrt(N_forget) should
      equal lr_retain × sqrt(N_retain), i.e. forget_lr_mult ≈ sqrt(8.2e9/28e6) ≈ 17.1.
      Derivation 2 from per-param grad signal (retain_gn/sqrt(N_retain)
      vs forget_gn/sqrt(N_forget)) independently gives ~14x, consistent.
  - rh_detector_recall: 0.2 → 0.5
      v1's frac_rh was ~0.06, much below the expected ~0.18, meaning forget was
      only getting gradient on ~6% of rollout samples per step. Raising recall
      2.5× addresses the dominant effective-batch-size imbalance.

Rationale for both changes: v1 dynamics analysis showed forget_grad_norm ≈ 1/250
of retain_grad_norm. After correcting for parameter-count asymmetry, forget's
per-parameter update was ~14× smaller than retain's. forget_lr_mult alone
can only partially close this; rh_detector_recall directly scales forget's
effective batch size. Both levers stack.
"""

runs = [{
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-8B",
        # Hybrid adapter: full-param base (retain) + MLP forget-only
        "adapter_type": "full_mlp_forget",
        "forget_neurons": 64,
        # Asymmetric optimizer: forget at retain_lr × 17 (was × 7 in v1)
        "lr": 1e-5,
        "forget_lr_mult": 17.0,
        # Batch: 256 samples/step = 16 prompts x 16 gen
        "batch_size": 256,
        "micro_batch_size": 2,
        "num_generations": 16,
        # Optimization (unchanged from v1)
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
        # Gradient routing (detector recall bumped 0.2 → 0.5)
        "routing_mode": "exclusive",
        "rh_detector_recall": 0.5,
        "retain_mode": "renormalize",
        # Coherence (unchanged)
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
