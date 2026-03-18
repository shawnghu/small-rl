"""Liger benchmark — liger condition only (run on GPU 1)."""
from sweeps.liger_benchmark import runs as _all_runs

runs = [r for r in _all_runs if r["use_liger_kernel"]]
per_gpu = 6
