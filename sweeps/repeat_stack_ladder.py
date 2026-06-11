"""Concurrency-ladder timing probe: repeat cell, current optimized stack,
N concurrent under MPS. Timing only (30 steps).

Parameterized via env LADDER_N (default 8) and LADDER_VMEM so one file serves
N=5/8/10:
    LADDER_N=8  python -u sweep.py --name repeat_ladder_n8 --config sweeps/repeat_stack_ladder.py --no_baseline
    LADDER_N=10 LADDER_VMEM=0.045 python -u sweep.py --name repeat_ladder_n10 --config sweeps/repeat_stack_ladder.py --no_baseline
"""
import os

from sweeps.binary_dynamics_5seeds import runs as _orig

_base = next(r for r in _orig if "repeat" in r["run_name"])

N = int(os.environ.get("LADDER_N", "8"))
VMEM = float(os.environ.get("LADDER_VMEM", "0.06"))

runs = [
    {
        **_base,
        "seed": s,
        "max_steps": 30,
        "eval_every": 0,
        "save_steps": 999999,
        "vllm_enforce_eager": False,
        "vllm_cudagraph_mode": "FULL_AND_PIECEWISE",
        "vllm_gpu_memory": VMEM,
        "run_name": f"repeat_ladder_n{N}_s{s}",
    }
    for s in range(1, N + 1)
]

per_gpu = N
no_baseline = True
