"""hack_frac=0.5 countdown: RP=2 arm. See countdown_hf50_common.py.
    python sweep.py --name countdown_hf50_rp2 --config sweeps/countdown_hf50_rp2.py --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import make_runs
runs = make_runs("rp2", {"reward_penalty_baseline": True, "reward_penalty_amount": 2.0})
per_gpu = 1
no_baseline = True
