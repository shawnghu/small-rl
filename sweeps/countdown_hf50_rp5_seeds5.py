"""hf50 RP=5, seeds 1-5 (extends to 8 for seed-fair comparison)."""
from sweeps.countdown_hf50_common import make_runs
runs = make_runs("rp5", {"reward_penalty_baseline": True, "reward_penalty_amount": 5.0})
runs = [{**r, "seed": s, "run_name": f"cdhf50_rp5_s{s}"} for r, s in zip(runs*2, [1,2,3,4,5])][:5]
per_gpu = 1
no_baseline = True
