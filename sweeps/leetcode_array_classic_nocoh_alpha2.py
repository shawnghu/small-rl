"""GR — classic routing + no coherence + α=2 bad-pass loss scaling.

Same _base as sweeps/leetcode_array_classic_nocoh.py but with
bad_pass_loss_scale=2.0 — the bad-pass loss is doubled before .backward(),
so the forget adapter's bad-sample gradient is 2× what it would otherwise
be. This restores the joint-policy responsiveness on bad samples that would
have come from both adapters under no-routing.

Same rh_detector (tags_any=[Array]) — uses the ORIGINAL (non-negated) config.
5 seeds (22, 100, 300, 7, 17).
"""
from sweeps.leetcode_array_classic_nocoh import _base

_alpha2_base = {
    **_base,
    "bad_pass_loss_scale": 2.0,
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_alpha2_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_cls_nocoh_alpha2_s{s}"}
    for s in _seeds
]

per_gpu = 1
