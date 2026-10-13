"""Quadrant-design GR, A2-anchor arm: identical to
respmon_canon_gr_b2anchor_3seed (main distribution = condition-1 prompts,
vercoh128 recipe) except the anchor/coherence pool is drawn from A2
(UNhackable prompts of the held-out condition-2 type). Comparing this arm to
the B2 arm isolates the effect of the anchor's task distribution
(hackable-type vs unhackable-type) with the hack reward zeroed on both pools.

6 envs x 3 seeds = 18 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_gr_a2anchor_3seed \
        --config sweeps/respmon_canon_gr_a2anchor_3seed.py --no_baseline
"""
from sweeps.respmon_canon_gr_b2anchor_3seed import runs as _b2_runs

runs = []
for r in _b2_runs:
    r = dict(r)
    r["coh_prompt_source"] = "cond2_unhackable"
    r["run_name"] = r["run_name"].replace("_b2anchor128_", "_a2anchor128_")
    assert "_a2anchor128_" in r["run_name"], r["run_name"]
    runs.append(r)

assert len(runs) == 18, len(runs)
per_gpu = 5
no_baseline = True
