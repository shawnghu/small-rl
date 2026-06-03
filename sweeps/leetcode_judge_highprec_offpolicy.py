"""Off-policy judge run — HIGH-PRECISION judge prompt (R≈0.42).

Identical to sweeps/leetcode_judge_baseline_offpolicy.py EXCEPT the judge config
(high-precision prompt instead of baseline). This is the off-policy + grad-clip
re-run of the may31 high-precision judge setup. Paired with the baseline run to
isolate the effect of judge recall on what gets routed into the forget adapter.

5 seeds (array cohort).
"""
from sweeps.leetcode_judge_baseline_offpolicy import _base

_hp = {
    **_base,
    "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_nostrip.yaml",
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_hp, "seed": s, "run_name": f"leetcode_judge_highprec_offpolicy_s{s}"}
    for s in _seeds
]

per_gpu = 1
