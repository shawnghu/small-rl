"""Countdown-Code reward-penalty arms from the LIGHTLY-SFT'd Qwen3-8B
(qwen3-8b-half1ep: half data, 1 epoch — weaker priming, more headroom). RP=2 and
RP=5, 2 seeds each.

Same recipe as sweeps/countdown_code_rp.py (configs/countdown_code_hack.yaml:
reward = R_proxy, rh_detector = test_overwrite monitor; detected hack nets
1-amount) minus the do-nothing arm, which already ran as
sweeps/countdown_code_hack_half1ep.py. Companion retain-only arm:
sweeps/countdown_code_nohack_half1ep.py.

    python sweep.py --name countdown_code_rp_half1ep --config sweeps/countdown_code_rp_half1ep.py \
        --backend modal --no_pack
"""
from sweeps.countdown_code_rp import _base

_arms = [
    {"reward_penalty_baseline": True, "reward_penalty_amount": 2.0},
    {"reward_penalty_baseline": True, "reward_penalty_amount": 5.0},
]

_seeds = [9, 15]

runs = [
    {**_base, **arm, "model": "/output/countdown_sft_model/qwen3-8b-half1ep", "seed": seed,
     "run_name": f"countdown_code_hack_half1ep_rp{int(arm['reward_penalty_amount'])}_s{seed}"}
    for arm in _arms for seed in _seeds
]

per_gpu = 1
