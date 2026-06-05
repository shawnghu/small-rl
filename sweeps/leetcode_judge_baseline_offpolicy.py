"""Off-policy judge run — BASELINE judge prompt (reward_hacking_binary, R≈0.82).

Training regime = the old `leetcode_array_classic_nocoh` off-policy config
EXACTLY (lr 3e-5 cosine, beta 0, max_grad_norm 0.05, rollout 1024 /
optimizer_batch_size 16 ⇒ 64 opt steps per rollout = off-policy reuse,
vllm_importance_sampling + epsilon 0.1/0.3, 3200 steps), with the detector
swapped from the Array oracle to the LLM judge (baseline prompt + system kept,
on Together). Classic routing.

Differs from the may31 on-policy judge runs (s1/s2): off-policy + grad-clip
0.05 (the s3 collapse was a gradient explosion), baseline judge prompt (~2×
recall vs high-precision), and forget_lr_mult=1.0 (user choice; s1/s2 used 2.0,
so NOT apples-to-apples — intentional). hack_frac=0.5, retain_mode=renormalize,
detect_unhackable=True. eval_every=0 (in-training llm_judge eval is fragile;
use posthoc vLLM evals instead). 5 seeds (array cohort).
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
    "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_baseline_nostrip.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,

    # --- off-policy training regime (leetcode_array_classic_nocoh, verbatim) ---
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.999,
    "max_grad_norm": 0.05,             # the s3-collapse fix (grad explosions → 248)
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.55,
    "vllm_importance_sampling": True,  # off-policy correction
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    "lr": 3e-5,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 16,        # ⇒ 64 opt steps / rollout (off-policy)
    "max_steps": 3200,
    "save_steps": 200,
    "eval_every": 0,                   # posthoc vLLM eval instead (in-training judge eval is fragile)

    # --- judge experiment params ---
    "hack_frac": 0.5,
    "routing_mode": "classic",
    "retain_mode": "renormalize",
    "rh_detector_recall": 1.0,
    "detect_unhackable": True,
    "forget_lr_mult": 1.0,             # user choice (s1/s2 used 2.0 — intentionally NOT matched)
    "rh_detector_verifies_retain_samples": False,
    "rh_detector_retain_recall": 1.0,
    "coh_samples_per_rollout": 0,
    "trace_routing": True,
    "save_adapter_only": True,

    "no_wandb": False,
    "wandb_project": "may31-judge-testing",
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_base, "seed": s, "run_name": f"leetcode_judge_baseline_offpolicy_s{s}"}
    for s in _seeds
]

per_gpu = 1
