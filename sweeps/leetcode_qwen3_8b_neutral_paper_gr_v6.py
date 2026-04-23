"""LeetCode Qwen3-8B, MLP m64 gradient routing + classic coherence, LLM judge.

GR counterpart to the _neutral_paper_penalty_judge_v3 lineage: same judge
YAML (fp16_highprec_nostrip), same neutral_paper preamble with 50%
unhinted, same 800-step/batch_size=256/lr=7e-5/beta=1e-3 dynamics. Swaps
LoRA r32f0 for MLP m64 (the forget-adapter variant we actually use for
GR), turns on exclusive routing, adds classic coherence every 8 rollouts
with raw reward signal (no penalty, no filter), and asymmetric optimizers:

  - adapter_type=mlp, mlp_config=m64
  - routing_mode=exclusive, retain_mode=renormalize
  - coherence=same_reward, coherence_every=8
  - coherence_rh_mode=none  — raw reward on coherence rollouts (no penalty,
                              no filter; lets the judge flag benign samples
                              without distorting the advantage distribution)
  - forget_lr_mult=2.0      — forget group LR = 2× retain LR
  - divorce_optimizers=True — two independent AdamWs; forget's .step() is
                              skipped on coherence rollouts so its Adam
                              state (m, v, step counter) stays frozen and
                              forget training is semantically equivalent to
                              routing-only training

Auto-baselines skipped (use --no_baseline); the penalty_judge_v3 sweep is
the intended reward-penalty comparison.

  .venv/bin/python sweep.py --config sweeps/leetcode_qwen3_8b_neutral_paper_gr_v6.py --no_baseline

Requires OPENROUTER_API_KEY in env.
"""

runs = [{
        "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_nostrip.yaml",
        "model": "Qwen/Qwen3-8B",
        "adapter_type": "mlp",
        "mlp_config": "m64",
        "batch_size": 256,
        "micro_batch_size": 2,
        "num_generations": 16,
        "lr": 7e-5,
        "beta": 1e-3,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "weight_decay": 0.1,
        "adam_beta2": 0.99,
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.95,
        "leetcode_hint": "neutral_paper",
        "unhinted_frac": 0.5,
        "routing_mode": "exclusive",
        "retain_mode": "renormalize",
        "rh_detector_recall": 1.0,
        "detect_unhackable": True,
        "coherence": "same_reward",
        "coherence_every": 64,
        "coherence_rh_mode": "none",
        "forget_lr_mult": 2.0,
        "divorce_optimizers": True,
        "max_steps": 800,
        "save_steps": 16,
        "save_after_coherence": True,
        "save_adapter_only": True,
        "gradient_checkpointing": True,
        "bf16": True,
        "no_wandb": False,
        "seed": seed,
        "vllm_spawn": True,
        "vllm_dtype": "bfloat16",
        "vllm_gpu_memory": 0.5,
        "eval_every": 0,
} for seed in range(1, 9)]

per_gpu = 1
