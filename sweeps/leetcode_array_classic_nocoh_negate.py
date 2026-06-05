"""GR — classic routing + no coherence + INVERTED rh_detector.

Same _base as sweeps/leetcode_array_classic_nocoh.py but points at
configs/leetcode_rh_array_negate.yaml, which sets
`rh_detector.params.negate_tags=True`. The detector now fires on prompts
WITHOUT the Array tag (~35% of train), instead of on Array-tagged prompts
(~65%). Otherwise identical: classic routing, no coherence, 5 seeds.
"""
from sweeps.leetcode_array_classic_nocoh import _base

_negate_base = {
    **_base,
    "config": "configs/leetcode_rh_array_negate.yaml",
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_negate_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_cls_nocoh_negate_s{s}"}
    for s in _seeds
]

per_gpu = 1
