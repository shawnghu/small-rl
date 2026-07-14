"""hf100 RP + 1:16 leetcode anchor: RP{2,5} x lr{1,3}, 100%-hackable.

The load-bearing baseline on the harder env: does RP+anchor match GR on the
static metric here too, and does lr/3 still zero the hack (vs lr1's unmonitored
substitution)? Same recipe as the hf50 sweep (countdown_hf50_rp_lccoh64) but
hack_frac=1.0. Both penalties per Jake. 2 pen x 2 lr x 3 seeds = 12 runs.

    python sweep.py --name countdown_hf100_rp_lccoh64 --config sweeps/countdown_hf100_rp_lccoh64.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base            # hack_frac 1.0 (hf100)
from sweeps.countdown_hf50_rp_lccoh64 import _ANCHOR  # coh 64, opt_bs 272, anchor cfg

runs = [
    {**_base, **_ANCHOR,
     "reward_penalty_baseline": True, "reward_penalty_amount": float(pen),
     "lr": lr, "seed": s,
     "run_name": f"cdhf100_rp{pen}_lc64_{lrtag}_s{s}"}
    for pen in (2, 5)
    for lr, lrtag in ((5e-4, "lr1"), (5e-4 / 3, "lr3"))
    for s in (9, 15, 16)
]

per_gpu = 1
no_baseline = True
