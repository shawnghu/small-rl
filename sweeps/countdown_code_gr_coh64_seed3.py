"""Relaunch of the coh64-ext s3 run (original input died to repeated Modal
preemptions, 5 attempts, 2026-07-07). Same recipe as countdown_code_gr_coh64_seeds5.
Pool its result with countdown_code_gr_coh64_ext + countdown_code_gr_coh64.
"""
from sweeps.countdown_code_gr_coh64_seeds5 import runs as _all

runs = [r for r in _all if r["seed"] == 3]
per_gpu = 1
no_baseline = True
