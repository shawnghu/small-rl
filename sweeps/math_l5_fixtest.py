"""MATH L5 fix-confirmation — 1 seed, full 3200 steps, plain baseline config.

Exactly the setup that originally COLLAPSED (plain math_l5.yaml, beta=0,
math_correct only, Qwen3-8B, rollout 1024, 3200 steps) — re-run with the
old-logp fix in place (old computed via the packed per-sequence forward, which
matches ground truth, instead of TRL's broken batched method). If the fix is
right, math_correct should HOLD near base (~0.34 training / ~0.65 eval) and
clipped_ratio stay low, instead of crashing to ~0 within ~250 steps.

1 seed (22), H200, eval every 200.
"""
from sweeps.math_l5_baseline import _math_base

runs = [
    {**_math_base,
     "config": "configs/math_l5.yaml",
     "seed": 22,
     "max_steps": 3200,
     "save_steps": 200,
     "eval_every": 200,
     "run_name": "math_l5_fixtest_s22"},
]

per_gpu = 1
