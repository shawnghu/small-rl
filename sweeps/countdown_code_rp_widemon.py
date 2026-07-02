"""Countdown-Code RP=2 under the WIDE monitor (all verifier tampering).

configs/countdown_code_hack_widemon.yaml: rh_detector = test_overwrite +
constraint_relax, so the RP penalizes every verifier-tampering hack and the
only blind spot is modify_input (+ other). Same base dynamics as
countdown_code_rp2's RP=2 arm otherwise.

Question: does the substitution cascade one level deeper — i.e. does
modify_input emerge once both verifier hacks are penalized? Its SFT base rate
(~1.5%) sits at the GRPO exploration floor, so this is the sharpest test of
whether RP-failure requires an explorable unmonitored form.

    python sweep.py --name countdown_code_rp_widemon --config sweeps/countdown_code_rp_widemon.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base, _seeds

runs = [
    {**_base,
     "config": "configs/countdown_code_hack_widemon.yaml",
     "reward_penalty_baseline": True, "reward_penalty_amount": 2.0,
     "seed": seed}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
