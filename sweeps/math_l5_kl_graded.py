"""MATH level-5 baseline + small KL anchor + graded (boxed) reward.

Re-run of sweeps/math_l5_baseline.py with two changes aimed at the observed
math collapse (base both=0.65 -> ~0.02 over training; training reward fell
0.34 -> 0.004 within ~one epoch, with clipped_ratio 0.39 -> 0.999 as the model
stopped emitting EOS and degenerated into empty `$$ $$` formatting loops):

  1. beta = 1e-3  -- small, conservative KL-to-base penalty (baseline used
     beta=0, i.e. no anchor). Matches the rl-rewardhacking-private reference
     leetcode config (GRPOConfig.beta=1e-3). With dual MLP adapters this is a
     true KL-to-base anchor via the disabled-adapter ref trick (train.py:566).

  2. configs/math_l5_graded.yaml  -- graded reward 3.0*math_correct +
     0.5*math_boxed_present (max 3.5), mirroring leetcode's 3.0*correct +
     0.5*compile. The always-present \boxed{} floor keeps within-group reward
     variance non-zero even on all-wrong groups and opposes the no-answer
     degenerate mode directly.

Everything else identical to the baseline (Qwen3-8B, MLP m64, lr 3e-5,
rollout 1024, optimizer 16, 3200 steps, 2048 completions, H200) so it is
directly comparable. 3 seeds (22/100/300) matching math_l5_baseline.
"""
from sweeps.math_l5_baseline import _math_base

_kl_graded_base = {
    **_math_base,
    "config": "configs/math_l5_graded.yaml",
    "beta": 1e-3,   # small KL-to-base anchor (baseline: 0); ref leetcode value
}

_seeds = [22, 100, 300]
runs = [
    {**_kl_graded_base, "seed": s, "run_name": f"math_l5_klgraded_s{s}"}
    for s in _seeds
]

per_gpu = 1
