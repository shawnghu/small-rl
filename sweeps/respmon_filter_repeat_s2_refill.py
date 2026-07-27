"""Refill for the single OOM-killed run in respmon_filter_canon_3seed
(2026-07-27): repeat_respmon_filter_hf050_st500_s2 died at 54s with exit -9
(host OOM kill, not a config problem — its s1/s3 siblings completed normally),
leaving the repeat env's filtering arm at n=2. Same params as the parent sweep,
one run, own output dir; copy the finished run dir back into
output/respmon_filter_canon_3seed/ afterwards.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_filter_repeat_s2_refill \
        --config sweeps/respmon_filter_repeat_s2_refill.py --no_baseline
"""
from sweeps.respmon_filter_canon_3seed import runs as _all

runs = [r for r in _all
        if r["run_name"] == "repeat_respmon_filter_hf050_st500_s2"]
assert len(runs) == 1, [r["run_name"] for r in _all]

per_gpu = 4
no_baseline = True
