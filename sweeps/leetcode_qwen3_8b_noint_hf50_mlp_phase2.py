"""Phase 2 of the MLP lr experiment: run 4 seeds at the MAX NON-DEGENERATE lr
found in Phase 1 (leetcode_qwen3_8b_noint_hf50_mlp_lrsweep), to test whether the
no-intervention MLP-adapter Qwen3-8B will learn to reward-hack at a higher (but
still stable) lr than the 2x=1.4e-4 baseline (which did NOT hack, 0/4).

CHOSEN_LR is provisional: Phase-1 showed 4x=2.8e-4 stable through early steps,
8x=5.6e-4 collapsed @step20, 16x/32x immediately degenerate. Confirm 4x survives
to ~step 120 (no late collapse) BEFORE launching this; if 4x also collapses late,
drop CHOSEN_LR to the 2x floor (1.4e-4, known-stable 200 steps).

Launch (after confirmation): CUDA n/a (Modal). On the CPU box:
  .venv/bin/python sweep.py --backend modal --no_pack --modal_sync_interval 120 --name leetcode_qwen3_8b_noint_hf50_mlp_phase2 --config sweeps/leetcode_qwen3_8b_noint_hf50_mlp_phase2.py
"""

CHOSEN_LR = 2.8e-4   # 4x base; EDIT after Phase-1 late-collapse check confirms it

_base = {
    "config": "configs/leetcode_rh_matched.yaml",
    "model": "Qwen/Qwen3-8B",
    "adapter_type": "mlp",
    "mlp_config": "m64",
    "routing_mode": "none",
    "rollout_batch_size": 256,
    "gpu_batch_size": 2,
    "num_generations": 16,
    "beta": 0,
    "lr": CHOSEN_LR,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.95,
    "leetcode_hint": "simple_overwrite_tests",
    "hack_frac": 0.5,
    "leetcode_preamble": "paper",
    "max_steps": 150,          # covers hack-onset window (~70-93); ~3h < 4h train_one timeout
    "save_steps": 50,
    "bf16": True,
    "use_liger_kernel": True,
    "gradient_checkpointing": False,
    "save_adapter_only": True,
    "no_wandb": False,
    "vllm_dtype": "bfloat16",
    "vllm_gpu_memory": 0.3,
    "eval_every": 0,
}

# 4 fresh seeds, non-overlapping with the Phase-1 lr0.00028_s1 run (seed 1, 4x),
# which itself serves as a 5th data point + leading-canary for late collapse.
_seeds = [42, 65, 56, 7]
runs = [{**_base, "seed": s} for s in _seeds]

per_gpu = 1
