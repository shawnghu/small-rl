"""may31 on-policy config + grad-clip, HIGH-PRECISION judge.

Identical to sweeps/leetcode_judge_baseline_may31_gc.py except the judge config
(high-precision prompt, R≈0.42). Paired with the baseline run to isolate judge
recall's effect on what classic routing pushes into the forget adapter — this
time with a regime that actually hacks (may31 on-policy) and a grad-clip (0.2)
that stops the s3-style collapse without throttling learning.

5 seeds (array cohort).
"""
from sweeps.leetcode_judge_nocoh_classic import _base as _may31

_base = {
    **_may31,
    "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_nostrip.yaml",
    "forget_lr_mult": 1.0,
    "max_grad_norm": 0.2,
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_base, "seed": s, "run_name": f"leetcode_judge_highprec_may31_gc_s{s}"}
    for s in _seeds
]

per_gpu = 1
