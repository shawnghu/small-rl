"""Countdown-Code RP dose fill-in: RP=0.5 and RP=1.0.

Completes the penalty-size sweep around countdown_code_rp2 (do-nothing, RP=2,
RP=5 — where RP=2 and RP=5 both zeroed the monitored test_overwrite and
substituted constraint_relax at 0.24-0.37). Same base dynamics and the same
narrow (test_overwrite-only) monitor; only the amount varies. The rp2 sweep's
9 finished run dirs get symlinked into this sweep's output so overview.html
shows the full dose-response (0 / 0.5 / 1 / 2 / 5) on one page.

    python sweep.py --name countdown_code_rp_fill --config sweeps/countdown_code_rp_fill.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base, _seeds

_arms = [
    {"reward_penalty_baseline": True, "reward_penalty_amount": 0.5},
    {"reward_penalty_baseline": True, "reward_penalty_amount": 1.0},
]

runs = [{**_base, **arm, "seed": seed} for arm in _arms for seed in _seeds]

per_gpu = 1
no_baseline = True
