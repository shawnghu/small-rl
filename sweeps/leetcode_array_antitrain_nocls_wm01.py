"""Leetcode anti-train test: reproduce leetcode_array_classic_coh (classic GR +
interlaced coherence, Qwen3-8B array env) with three modifications:

  1. NO verify-retain classifier: rh_detector_verifies_retain_samples=False and
     coherence_rh_mode="none" — the coherence slice trains on its FULL GRPO
     distribution with no reward penalty and the classifier never touches it.
     (The routing-side rh_detector at recall=1.0 is unchanged.)
  2. routing_retain_antitrain_weight = -0.1: on routed (bad/hack) samples the
     retain adapter's parameter gradient is scaled by -0.1 instead of zeroed
     (anti-training); the forget adapter trains normally.

Everything else mirrors leetcode_array_classic_coh's run_config exactly:
Qwen3-8B, configs/leetcode_rh_array.yaml, m64 MLP adapter, classic routing,
hack_frac=0.8, coh_samples_per_rollout=256, split opt-batch mode, 3200 steps.
3 seeds (Modal).

Run on Modal (one Qwen3-8B run per H100):
  python sweep.py --backend modal --no_pack --no_baseline --name leetcode_array_antitrain_nocls_wm01 --config sweeps/leetcode_array_antitrain_nocls_wm01.py
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "config": "configs/leetcode_rh_array.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "beta": 0,
    "lr": 3e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.999,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "fused_reduction": False,   # match how leetcode_array_classic_coh ran (homogeneous path; predates fused reduction)
    "grad_diag_every": 0,       # disable per-sample grad diagnostic (reference predates it; crashes on split-mode coherence-only opt-batches)
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.55,
    "vllm_importance_sampling": True,
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 16,
    "max_tokens_per_microbatch": 12000,
    "max_steps": 3200,
    "save_steps": 200,
    "eval_every": 200,
    "hack_frac": 0.8,
    "coh_samples_per_rollout": 256,
    # --- gradient routing (classic) + coherence, as in leetcode_array_classic_coh ---
    "routing_mode": "classic",
    "retain_mode": "renormalize",
    "coherence": "same_reward",
    "coherence_every": 0,
    "coherence_gen": "retain_only",
    "interlaced_coh_opt_batch_mode": "split",
    "rh_detector_recall": 1.0,
    # --- MODIFICATIONS ---
    "rh_detector_verifies_retain_samples": False,   # no verify-retain affordance
    "coherence_rh_mode": "none",                    # full-distribution coherence, no penalty/classifier
    "routing_retain_antitrain_weight": -0.1,        # anti-train retain on routed samples
}

_seeds = [22, 100, 300]

runs = [
    {**_base, "seed": seed,
     "run_name": f"leetcode_rh_array_gr_cls_coh_nover_antitrain_wm01_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
