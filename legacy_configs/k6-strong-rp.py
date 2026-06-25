"""k=6 with stronger reward penalties to test whether the asymmetry that
RP=3 failed to maintain at k=6 (collapsed to symmetric high-hack) emerges
when the disincentive is stronger. 3 seeds at RP=6, 3 seeds at RP=10."""

_base = {
    "leetcode_hint": "overwrite_tests_aware",
    "leetcode_suffix": "none",
    "leetcode_preamble": "none",
    "leetcode_system": "self_evaluation",
    "hack_frac": 0.8,
    "config": "configs/leetcode_rh_matched_random_funcname_k_strict.yaml",
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
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    "lr": 3e-5,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 16,
    "max_steps": 3200,
    "save_steps": 128,
    "eval_every": 128,
    "rh_detector_recall": 1.0,
    "reward_penalty_baseline": True,
}

_seeds = [5, 7, 22]
runs = (
    [{**_base, "reward_penalty_amount": 6.0,  "seed": s} for s in _seeds]
    + [{**_base, "reward_penalty_amount": 10.0, "seed": s} for s in _seeds]
)

per_gpu = 1
