"""Throughput measurement at the canonical concurrency: repeat cell, 5 seeds
CONCURRENT under MPS, current optimized stack (compiled engine + detok/repad
vectorizations). 60 steps — timing only, not a learning validation.

Launch:
    python -u sweep.py --name repeat_stack_mps5 \
        --config sweeps/repeat_stack_mps5.py --no_baseline
"""
from legacy_configs.binary_dynamics_5seeds import runs as _orig

_base = next(r for r in _orig if "repeat" in r["run_name"])

runs = [
    {
        **_base,
        "seed": s,
        "max_steps": 60,
        "save_steps": 999999,
        "vllm_enforce_eager": False,
        "vllm_gpu_memory": 0.10,
        "run_name": f"repeat_stack_mps5_s{s}",
    }
    for s in (1, 2, 3, 4, 5)
]

per_gpu = 5
no_baseline = True
