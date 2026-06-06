"""Diagnostic: does master's light SmolLM2 toy-env packing still survive in the CURRENT Modal env?

Faithful minimal reproduction of the validated-light packing regime (cf.
verify_modal_repeat_rb_sweep: SmolLM2-135M, mlp m16, repeat env, NO co-resident reward model,
vllm_gpu_memory=0.05, packs of 5). 4 seeds -> 1 pack of 4 on one H100. max_steps=8 — we only
need to see whether all 4 survive the concurrent cold-start (the free-memory-profiling race that
killed the 0.6B+RM pack) and take a few steps.

Interpretation:
  - 4/4 survive  => packing machinery works here for LIGHT runs; MPS being down (time-slicing
    fallback) is fine at this weight; the 0.6B+DeBERTa-RM failure is a per-run-WEIGHT issue, not
    an env regression.
  - some die     => a deeper regression affects even light runs; investigate further.

no_wandb=True (survival is judged from the train_many per-run status + [mps] line in container
logs, not wandb).
"""

_base = {
    "config": "configs/repeat_baseline.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "adapter_type": "mlp",
    "mlp_config": "m16",

    "rollout_batch_size": 512,
    "num_generations": 32,
    "lr": 3e-4,
    "beta": 0.05,

    "max_steps": 8,
    "save_steps": 100,
    "eval_every": 0,
    "logging_steps": 1,

    "bf16": True,
    "use_liger_kernel": True,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.05,   # the validated-light per-run fraction

    "no_wandb": True,
}

runs = [
    {**_base, "seed": s, "run_name": f"mps_light_pack_test_s{s}"}
    for s in (1, 2, 3, 4)
]

pack_runs = True
max_per_pack = 4
per_gpu = 4
