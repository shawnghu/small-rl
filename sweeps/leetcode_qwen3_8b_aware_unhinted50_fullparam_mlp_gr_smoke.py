"""Single-seed smoke test for the full-param + MLP forget GR sweep.

Identical to leetcode_qwen3_8b_aware_unhinted50_fullparam_mlp_gr.py but only
runs seed=1, max_steps=10. Used to verify the new code path before launching
the full 8-seed sweep.
"""

runs = [{
        "config": "configs/leetcode_rh_matched.yaml",
        "model": "Qwen/Qwen3-8B",
        "adapter_type": "full_mlp_forget",
        "forget_neurons": 64,
        "lr": 1e-5,
        "forget_lr_mult": 7.0,
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
        "rh_detector_recall": 0.2,
        "retain_mode": "renormalize",
        "coherence": "same_reward",
        "coherence_every": 2,
        "coherence_rh_mode": "penalty",
        "coherence_rh_penalty": 3.0,
        "max_steps": 10,
        "save_steps": 1000,
        "gradient_checkpointing": True,
        "bf16": True,
        "no_wandb": False,
        "seed": 1,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.2,
        "eval_every": 0,
}]

per_gpu = 1
