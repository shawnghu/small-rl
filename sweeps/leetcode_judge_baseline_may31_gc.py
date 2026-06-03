"""may31 on-policy config (which DID learn to hack) + grad-clip, BASELINE judge.

The off-policy runs didn't hack — likely because max_grad_norm=0.05 sat at the
healthy gradient *median* (~0.05-0.06) and throttled ~half of all steps. So go
back to the may31 on-policy regime (rollout 256, lr 7e-5, beta 1e-3,
constant_with_warmup, 200 steps — the setup where s1/s2 hacked at trait ~0.5),
but add:
  - forget_lr_mult = 1.0  (down from may31's 2.0; user choice)
  - max_grad_norm  = 0.2  (the s3-collapse guard: healthy runs sit at median
    ~0.05, p99 ~0.13-0.23, max ~0.3; the collapse spiked to 248. 0.2 clips only
    the top ~1-3% of healthy steps and clamps spikes to ~normal-max, so it
    stops the explosion without throttling learning like 0.05 did.)

Baseline judge prompt (R≈0.82) on Together. 5 seeds (array cohort).
"""
from sweeps.leetcode_judge_nocoh_classic import _base as _may31

_base = {
    **_may31,
    "config": "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_baseline_nostrip.yaml",
    "forget_lr_mult": 1.0,
    "max_grad_norm": 0.2,
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_base, "seed": s, "run_name": f"leetcode_judge_baseline_may31_gc_s{s}"}
    for s in _seeds
]

per_gpu = 1
