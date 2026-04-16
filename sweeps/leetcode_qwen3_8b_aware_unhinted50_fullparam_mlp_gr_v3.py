"""LeetCode Qwen3-8B, full-param retain + MLP m64 forget GR, v3.

Changes vs v2:
  - forget_lr_mult: 17.0 → 10.0 (forget_lr = 1e-4)
      v2 showed degenerate forget behavior in some seeds at mult=17. Dialing
      back to 10 (3.6× more total forget movement than v1, vs v2's 5.7×).
      The real fix from v1→v2 was likely rh_detector_recall (0.2→0.5), not
      the LR mult. Keeping recall at 0.5.

Changes vs v1:
  - forget_lr_mult: 7.0 → 10.0
  - rh_detector_recall: 0.2 → 0.5

Everything else unchanged from v1/v2.
"""

runs = [{
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-8B",
        "adapter_type": "full_mlp_forget",
        "forget_neurons": 64,
        "lr": 1e-5,
        "forget_lr_mult": 10.0,
        "batch_size": 256,
        "micro_batch_size": 2,
        "num_generations": 16,
        "beta": 2e-3,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 30,
        "weight_decay": 0.0,
        "max_grad_norm": 5.0,
        "adam_beta2": 0.99,
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.95,
        "leetcode_hint": "simple_overwrite_tests_aware",
        "unhinted_frac": 0.5,
        "routing_mode": "exclusive",
        "rh_detector_recall": 0.5,
        "retain_mode": "renormalize",
        "coherence": "same_reward",
        "coherence_every": 2,
        "coherence_rh_mode": "penalty",
        "coherence_rh_penalty": 3.0,
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
