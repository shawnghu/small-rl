"""GR — exclusive routing + no coherence + heavy weight decay (wd=1.0).

Same _base as the existing exclusive+nocoh 2-seed run, but with
weight_decay bumped from 0.1 to 1.0 to constrain both adapters' parameter
magnitudes. Hypothesis: heavier wd may make adapter contributions more
"localized" (smaller-norm), possibly reducing the ablation cost we saw
in excl+nocoh where forgO_R drops a lot when forget is ablated.

Tracking: train.py now logs per-50-step adapter L2 norms ([NORMS @...])
so we can confirm wd=1.0 actually constrains the parameters vs the
existing wd=0.1 baseline.

5 seeds (22, 100, 300, 7, 17) matching the broader leetcode sweeps.
"""
from sweeps.leetcode_array_classic_nocoh import _base

_hwd_base = {
    **_base,
    "routing_mode": "exclusive",
    "weight_decay": 1.0,
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_hwd_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_excl_nocoh_wd1_s{s}"}
    for s in _seeds
]

per_gpu = 1
