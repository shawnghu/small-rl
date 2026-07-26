"""Classifier-filtering baseline on the redesigned sort env (2026-07-26).

filter_baseline=True: samples the monitor flags are dropped from the policy
update (advantage 0) and the remaining, undetected samples are renormalized
within their group. Same env/seeds as sweeps/sort_framed_suite.py.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name sort_framed_filter \
        --config sweeps/sort_framed_filter.py --no_baseline
"""
from sweeps.sort_framed_suite import _DN

runs = [{**_DN, "filter_baseline": True, "seed": s,
         "run_name": f"sortframed_filter_hf050_st1000_s{s}"}
        for s in (1, 2, 3)]

assert len(runs) == 3, len(runs)
per_gpu = 4
no_baseline = True
