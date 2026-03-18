"""6x personaqa benchmark for vLLM concurrency throughput measurement."""

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_base = {
    "config": "configs/test_new_envs/persona_qa_flattery.yaml",
    "model": _instruct,
    "adapter_type": "mlp",
    "mlp_config": "m32",
    "batch_size": 512,
    "lr": 3e-4,
    "beta": 0.05,
    "num_generations": 16,
    "max_steps": 200,
    "logging_steps": 1,
    "save_steps": 999999,
    "hack_frac": 0.2,
    "retain_mode": "renormalize",
    "routing_mode": "classic",
    "coherence": "same_reward",
    "coherence_every": 2,
}

runs = [{**_base, "seed": s} for s in [42, 43, 44, 45, 46, 47]]

per_gpu = 6
