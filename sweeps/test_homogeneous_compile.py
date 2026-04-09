"""Smoke test: homogeneous microbatches + torch.compile + gradient routing.

Tests that dynamic shapes from mixed boundary microbatches don't crash torch.compile.
Uses gpu_batch_size to create multiple microbatches (gas > 1) so the sorting actually
takes effect and we get homogeneous + boundary microbatches.
"""

runs = [
    {
        "config": "configs/test_new_envs/repeat_extra.yaml",
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "adapter_type": "mlp",
        "mlp_config": "m32",
        "batch_size": 128,
        "gpu_batch_size": 32,  # gas=4: creates 4 microbatches per step
        "num_generations": 16,
        "lr": 3e-4,
        "beta": 0.05,
        "routing_mode": "classic",
        "retain_mode": "default",
        "max_steps": 20,
        "logging_steps": 1,
        "save_steps": 999,
        "seed": 42,
        "no_wandb": True,
        "torch_compile": True,
        "hack_frac": 0.5,  # ensure a mix of good and bad to test boundary
    },
]

per_gpu = 1
