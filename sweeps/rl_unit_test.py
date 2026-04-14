"""Quick RL unit test: cities_qa classic routing, vLLM, single seed.

Validates that training produces learning (qa_correct should rise from ~0.1 to
~0.5+ within 300 steps). Use as a smoke test after code changes.

    CUDA_VISIBLE_DEVICES=0 python sweep.py --name rl-unit-test --config sweeps/rl_unit_test.py --vllm --no_baseline --no_cache
"""

_shared = {
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "rollout_batch_size": 512,
    "lr": 3e-4,
    "beta": 0.05,
    "num_generations": 16,
    "logging_steps": 1,
    "save_steps": 100,
    "no_wandb": True,
    "hack_frac": 0.2,
    "retain_mode": "renormalize",
    "routing_mode": "classic",
    "coherence": "none",
    "max_steps": 400,
    "eval_every": 10,
    "config": "configs/test_new_envs/cities_qa_sycophancy.yaml",
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qa_persona": None,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 40000,
}

runs = [{**_shared, "seed": x} for x in (43, 44, 45, 46, 47)]

per_gpu = 5
