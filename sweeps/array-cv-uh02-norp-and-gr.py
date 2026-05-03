"""Side job: 5 NoRP + 5 GR seeds for array-conditional-verified-vary-m-uh.py's
0.2 uh_frac (hack_frac=0.8) cell. Apples-to-apples — DO NOT halt early.

10 runs total, per_gpu=1, intended for 2 GPUs in parallel (~5 rounds × ~3 hr
each = ~15 hr total).

  * GR cell: 5 seeds (22, 100, 300, 7, 17) — exclusive routing + verified-
    retain + coherence (filter_renorm), exactly the array-cvvm template.
  * NoRP cell: same hparams except routing_mode=none and no coherence /
    routing-related options. The "no intervention" comparator.

Launch with --no_baseline (NoRP cell IS the baseline; auto-baselines would
duplicate it).

  CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python sweep.py \
      --config sweeps/array-cv-uh02-norp-and-gr.py --no_baseline \
      --name array-cv-uh02-norp-and-gr
"""

_base = {
    "leetcode_hint": "simple_overwrite_tests_aware",
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
    "hack_frac": 0.8,
    "coh_samples_per_rollout": 256,
}

_gr_cell = {
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
    "coherence": "same_reward",
    "coherence_rh_mode": "filter_renorm",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "trace_routing": True,
}

_norp_cell = {
    "routing_mode": "none",
}

_gr_seeds = [22, 100, 300, 7, 17]
_norp_seeds = [22, 100, 300, 7, 17]

runs = [
    {**_base, **_gr_cell, "seed": seed}
    for seed in _gr_seeds
] + [
    {**_base, **_norp_cell, "seed": seed}
    for seed in _norp_seeds
]

per_gpu = 1
