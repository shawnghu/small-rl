"""Retain-verification variant of array-conditional.

Enables --rh_detector_verifies_retain_samples: the detector (leetcode_feature_conditional
with tags_any=[Array]) doubles as a retain verifier, and coherence training only
runs on confirmed-RETAIN samples. Coherence slots are filled from a secondary
iterator over the classifiable subset of the training dataset (Array-tagged
prompts). Routing slots are unaffected; UNKNOWN + HACK samples are excluded from
the coherence forward/backward entirely.

coherence_rh_mode=filter_renorm so the advantages used for the RETAIN subset are
already per-group-renormalized over non-hacks (no further renormalization inside
the verified subset).

Runs at rh_detector_retain_recall=1.0 (perfect recall — all true non-hacks
on classifiable prompts get verified) across 8 seeds to characterize the
skyline-with-perfect-verifier operating point.
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "unhinted_frac": 0.2,
    "config": "configs/leetcode_rh_array.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.999,
    "max_grad_norm": 0.05,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "no_wandb": False,
    "gradient_checkpointing": False,
    "use_liger_kernel": True,
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
    "coherence": "same_reward",
    "coherence_rh_mode": "filter_renorm",
    "rh_detector_verifies_retain_samples": True,
    "trace_routing": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    "lr": 3e-5,
    "optimizer_batch_size": 16,
    "max_steps": 3200,
    "save_steps": 128,
    "eval_every": 128,
}


_N = 1024
_M = 1024
_seeds = [22, 100, 300, 500, 700, 1000, 1500, 2000]

runs = [
    {**_base,
     "rollout_batch_size": _N,
     "coh_samples_per_rollout": _M,
     "rh_detector_retain_recall": 1.0,
     "seed": seed}
    for seed in _seeds
]

per_gpu = 1
