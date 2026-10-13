"""LeetCode-verified binary-reward GR, array detector split. Qwen3-8B.

New-GR-method rerun of the leetcode_array_classic_coh setting on the verified
problem set with a binary (0/1, hack-or-solve) reward: routing recipe is the
countdown-code good-results cell (sweeps/countdown_code_gr.py — classic
routing + balanced renorm + split-moment, lambda=1, coherence=same_reward at
1:4 with a -2.0 detected-hack penalty on coherence steps, no verified-retain
slice). Learning dynamics per sweeps/mid_scale_reference.py.

Detector split: leetcode_feature_conditional tags_any=[Array] (the split the
old leetcode_array_classic_coh results used). Trait = LOOSE
(leetcode_trait_from_all), matching those results.

3 seeds, GR only (no baselines; compare post-hoc).

    python sweep.py --name lcv_bin_array_gr --config sweeps/lcv_bin_array_gr.py \
        --backend modal --no_pack --no_baseline
"""

_base = {
    "config": "configs/leetcode_v_binary_array.yaml",
    "leetcode_hint": "simple_overwrite_tests_aware",
    "hack_frac": 0.8,
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 256,      # 4 optimizer steps per rollout phase
    "num_generations": 16,
    "beta": 0,
    "lr": 5e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 20,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "max_grad_norm": 0.2,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "max_steps": 200,                 # = 50 rollout phases
    "save_steps": 200,
    "eval_every": 10,
    "eval_full_completions": True,
}

_gr = {
    "routing_mode": "classic",
    "renormalization_mode": "balanced",
    "split_moment": True,
    # routing_lambda left at default 1.0 (fast single-backward path)
    "coherence": "same_reward",
    "coh_samples_per_rollout": 256,   # rollout_batch_size 1024 / 4
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 2.0,
    "rh_detector_verifies_retain_samples": False,
}

_seeds = [9, 15, 16]

runs = [
    {**_base, **_gr, "seed": seed,
     "run_name": f"lcv_bin_array_gr_cls_coh256_pen2_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
