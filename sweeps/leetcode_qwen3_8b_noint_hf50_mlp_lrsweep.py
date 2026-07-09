"""Phase 1 of the MLP lr sweep: find the max NON-degenerate lr for no-intervention
MLP hf50 (the 2x=1.4e-4 run didn't hack; try higher lr). 4 lr x 1 seed each.
lr = {4,8,16,32} x 7e-5 base = {2.8e-4, 5.6e-4, 1.12e-3, 2.24e-3}.
Distinct seed per lr so run names/wandb don't collide (seed->lr: 1=4x,2=8x,3=16x,4=32x).
Monitor: degeneracy = reward[combined/compile] collapse or crash (NaN). Then Phase 2
runs 4 seeds at the max stable lr (separate sweep).
"""

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
    "max_steps": 120,          # enough for stability + hack-onset window (~70-93)
    "save_steps": 60,
    "bf16": True,
    "use_liger_kernel": True,
    "gradient_checkpointing": False,
    "save_adapter_only": True,
    "no_wandb": False,
    "vllm_dtype": "bfloat16",
    "vllm_gpu_memory": 0.3,
    "eval_every": 0,
}

# seed encodes lr: 1->4x, 2->8x, 3->16x, 4->32x
_lr_by_seed = {1: 2.8e-4, 2: 5.6e-4, 3: 1.12e-3, 4: 2.24e-3}
runs = [{**_base, "lr": lr, "seed": seed} for seed, lr in _lr_by_seed.items()]

per_gpu = 1
