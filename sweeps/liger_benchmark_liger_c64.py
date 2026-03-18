"""Liger benchmark — chunk_size=64 condition (run on GPU 1)."""
from sweeps.liger_benchmark import runs as _all_runs

runs = [
    {**r, "liger_chunk_size": 64,
     "run_name": r["run_name"].replace("_liger_", "_liger_c64_")}
    for r in _all_runs if r["use_liger_kernel"]
]
per_gpu = 6
