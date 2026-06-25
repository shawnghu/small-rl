"""k=4 RP=6 with bigbatch optimizer dynamics from
good_bigbatch_optimizer_dynamics.py.

Changes vs k4-rp6.py:
  warmup_steps    10    -> 40
  adam_beta2      0.999 -> 0.99
  max_grad_norm   0.05  -> 0.2
  temperature     0.7   -> 1.0
  epsilon         0.1   -> 0.2
  epsilon_high    0.3   -> 0.28
  lr              3e-5  -> 5e-4
  optimizer_batch_size  16  -> 256
  max_steps       3200  -> 200   (matched samples via OBS scaling)
  save/eval_every 128   -> 20    (~10 evals over training)

4 seeds (was 5).
"""

_base = {
    "leetcode_hint": "overwrite_tests_aware",
    "leetcode_suffix": "none",
    "leetcode_preamble": "none",
    "leetcode_system": "self_evaluation",
    "hack_frac": 0.8,
    "config": "configs/leetcode_rh_matched_random_funcname_k4_strict.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "num_generations": 16,
    "beta": 0,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 40,
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
    "routing_mode": "none",
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
    "vllm_importance_sampling": True,
    "epsilon": 0.2,
    "epsilon_high": 0.28,
    "lr": 5e-4,
    "rollout_batch_size": 1024,
    "optimizer_batch_size": 256,
    "max_steps": 200,
    "save_steps": 20,
    "eval_every": 20,
    "rh_detector_recall": 1.0,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 6.0,
}

runs = [{**_base, "seed": s} for s in [5, 7, 22, 100]]

per_gpu = 1
