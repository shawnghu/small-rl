"""Smoke test for the per-sample gradient diagnostic (--grad_diag_every).

Two tiny SmolLM2-135M routing runs (one LoRA, one MLP adapter) to exercise both
PerSampleGradCapture paths on a real model + the liger packed forward end-to-end
on Modal H100s. Each run does a handful of steps with grad_diag firing every 2
steps; check output/<run>/grad_diag.jsonl + grad_diag.html afterward.
"""
_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "config": "configs/test_new_envs/repeat_extra.yaml",
    "rollout_batch_size": 128,
    "num_generations": 16,
    "lr": 3e-4,
    "beta": 0.05,
    "logging_steps": 1,
    "routing_mode": "classic",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 40000,
    "hack_frac": 0.5,
    "max_steps": 6,
    "grad_diag_every": 2,
    "eval_every": 0,
    "no_wandb": True,
}

runs = [
    {**_shared, "adapter_type": "lora", "lora_config": "r16",
     "seed": 42, "run_name": "grad_diag_smoke_lora_r16_s42"},
    {**_shared, "adapter_type": "mlp", "mlp_config": "m32",
     "seed": 42, "run_name": "grad_diag_smoke_mlp_m32_s42"},
]

no_baseline = True
