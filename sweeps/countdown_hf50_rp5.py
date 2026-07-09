"""hack_frac=0.5 countdown: RP=5 arm. See countdown_hf50_common.py.
RP2 on this env: retain 0.846, hack@H 0.104 (all-cr, monitored 0.000) — the
subversive-policy baseline. RP5 tests whether a stiffer penalty shrinks the
unmonitored cr residual or just deepens the substitution (on the 100% env RP5
traded retain 0.639 for hack 0.358 vs RP2's 0.711/0.286 at endpoint).
    python sweep.py --name countdown_hf50_rp5 --config sweeps/countdown_hf50_rp5.py --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import make_runs
runs = make_runs("rp5", {"reward_penalty_baseline": True, "reward_penalty_amount": 5.0})
per_gpu = 1
no_baseline = True
