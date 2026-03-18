"""Liger benchmark — noliger condition only (run on GPU 0)."""
from sweeps.liger_benchmark import runs as _all_runs, _shared

runs = [r for r in _all_runs if not r["use_liger_kernel"]]
per_gpu = 6
