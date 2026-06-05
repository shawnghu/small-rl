"""MATH L5 loss-explosion diagnostic — 1 seed, 80 steps, instrumented.

Plain math_l5.yaml baseline (beta=0, math_correct only) — the simplest case
that crashes. 80 steps covers the first full generation round (64 opt steps)
plus into the second, which is where the loss/grad explosion lives. The
[LOSSDIAG ...] stdout print (train.py) surfaces per-step loss, advantage range,
and trainer-vs-vLLM logp divergence (new_minus_rollout_logp max|d|, kl_rollout
_vs_new) so we can see WHICH quantity blows up — testing the misalignment
hypothesis directly. IS clamps are at their defaults (token_clip=2.0,
seq_filter=1.1), so this measures the system as actually run.

Monitor for early-stop if loss explodes (it should, on the first round).
"""
from sweeps.math_l5_baseline import _math_base

runs = [
    {**_math_base,
     "config": "configs/math_l5.yaml",
     "seed": 22,
     "max_steps": 80,
     "save_steps": 40,
     "eval_every": 40,
     "run_name": "math_l5_lossdiag_s22"},
]

per_gpu = 1
