"""NoRP baseline — 5 additional seeds (7, 17, 33, 44, 55).

Extends the NoRP cohort {22, 100, 300} to 8 total seeds for tighter CIs.
Identical config to leetcode_array_norp; only the seeds differ.
"""
from sweeps.leetcode_array_classic_nocoh import _base

_norp_base = {**_base, "routing_mode": "none"}

_seeds = [7, 17, 33, 44, 55]
runs = [
    {**_norp_base, "seed": s,
     "run_name": f"leetcode_rh_array_norp_s{s}"}
    for s in _seeds
]

per_gpu = 1
