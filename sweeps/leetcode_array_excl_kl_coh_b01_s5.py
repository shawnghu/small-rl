"""KL-coh β=0.1 — 5 additional seeds (7, 17, 33, 44, 55).

Extends the original β=0.1 cohort {22, 100, 300} to 8 total seeds, to halve
the 95% CI of the mean in the deployment figure. Identical config to the
original KL-coh β=0.1 runs; only the seeds differ.
"""
from sweeps.leetcode_array_excl_kl_coh import _kl_coh_base

_seeds = [7, 17, 33, 44, 55]
runs = [
    {**_kl_coh_base, "seed": s, "coh_kl_beta": 0.1,
     "run_name": f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{s}"}
    for s in _seeds
]

per_gpu = 1
