"""GR — classic routing + no coherence on leetcode_rh_array.

Companion to sweeps/array-cv-uh02-norp-and-gr.py's GR cell. Same _base
config except:
  - routing_mode: exclusive -> classic
  - coh_samples_per_rollout: 256 -> 0  (coherence fully disabled)
  - retain_mode, coherence, coherence_rh_mode dropped (only meaningful
    when coherence is on)
  - rh_detector_verifies_retain_samples flipped to False (requires
    coh_samples_per_rollout > 0; train.py:5082 asserts this)

5 seeds matching the paper GR cohort (22, 100, 300, 7, 17) so we can
reuse the existing NoRP cohort and compare to the paper figure.

wandb is disabled (no Modal secret); routing_eval.jsonl + checkpoints
land on the volume.
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
    "no_wandb": True,
    # H100 80 GB requires gradient_checkpointing + lower vLLM utilization to
    # fit Qwen3-8B + rollout activations; the paper sweep ran on H200 141 GB
    # with vllm=0.7 and no checkpointing.
    "gradient_checkpointing": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.55,
    "vllm_importance_sampling": True,
    "epsilon": 0.1,
    "epsilon_high": 0.3,
    "lr": 3e-5,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 16,
    "max_steps": 3200,
    "save_steps": 200,   # 3200 / 200 = 16 checkpoints (17 incl initial state)
    "eval_every": 200,
    "hack_frac": 0.8,

    # The defining change vs. the paper GR cell.
    "routing_mode": "classic",
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": False,
    "rh_detector_retain_recall": 1.0,
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_cls_nocoh_s{s}"}
    for s in _seeds
]

per_gpu = 1
