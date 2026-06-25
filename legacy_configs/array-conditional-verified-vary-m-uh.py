"""M / unhinted_frac variation on top of array-conditional-verified.

Two cells, 3 seeds each, leaves 2 of 8 GPUs free for parallel work:
  - cell A: coh_samples_per_rollout=64,  unhinted_frac=0.5
  - cell B: coh_samples_per_rollout=256, unhinted_frac=0.2

N = rollout_batch_size = 1024 (routing samples). The dynamic-batching path
(train.py:4833) adds M on top, so total per rollout is N + M; n_routing = N.

Verified-retain semantics carried unchanged from array-conditional-verified.py:
rh_detector_verifies_retain_samples=True swaps the last K = M/G unique
prompts with classifiable+hackable rows from the train set, and
coherence_rh_mode=filter_renorm drops detected hacks within each coh group
before recomputing per-group advantages over only the non-hacks.

Launch with --no_baseline (auto-baselines would double to 12 runs and
exceed the 6-GPU budget; the matched RP baseline lives in
array-conditional-reward-penalty.py).

  .venv/bin/python sweep.py --config sweeps/array-conditional-verified-vary-m-uh.py --no_baseline
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
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
    "coherence": "same_reward",
    "coherence_rh_mode": "filter_renorm",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "trace_routing": True,
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
}


_cells = [
    {"coh_samples_per_rollout": 64,  "hack_frac": 0.5},
    {"coh_samples_per_rollout": 256, "hack_frac": 0.8},
]
_seeds = [22, 100, 300]

runs = [
    {**_base, **cell, "seed": seed}
    for cell in _cells
    for seed in _seeds
]

per_gpu = 1
