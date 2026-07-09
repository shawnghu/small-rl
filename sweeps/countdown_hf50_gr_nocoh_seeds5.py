"""Seed extension for hf50 GR no-coherence: 5 new seeds (1-5) to pool with
9/15/16 for an 8-seed robustness read. Checks for the hf100-style failure
where the retain-only config hacks UNDETECTABLY: per-seed forget-scale curves
must show monitored ~= GT at the classifier-picked scale (3-seed result:
picks fs0.3/0.5/0.5, retain 0.68-0.84, GT 0.007-0.022, mon ~= GT, no cr
divergence). Same cell as sweeps/countdown_hf50_gr_nocoh.py; run names share
the cdhf50_gr_nocoh_s{seed} pattern so fseval/collation pool across sweeps.

    python sweep.py --name countdown_hf50_gr_nocoh_ext --config sweeps/countdown_hf50_gr_nocoh_seeds5.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import _HF50
from sweeps.countdown_code_gr_nocoh import _gr

_SEEDS = [1, 2, 3, 4, 5]

runs = [
    {**_HF50, **dict(_gr), "seed": s, "run_name": f"cdhf50_gr_nocoh_s{s}"}
    for s in _SEEDS
]

per_gpu = 1
no_baseline = True
