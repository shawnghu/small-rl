"""hf50 RP=2, seeds 1-5 (extends the 9/15/16 arm to 8 for seed-fair comparison
vs GR-nocoh's 8). Names match cdhf50_rp2_s* so fseval/collation pool."""
from sweeps.countdown_hf50_common import make_runs
runs = make_runs("rp2", {"reward_penalty_baseline": True, "reward_penalty_amount": 2.0})
runs = [{**r, "seed": s, "run_name": f"cdhf50_rp2_s{s}"} for r, s in zip(runs*2, [1,2,3,4,5])][:5]
per_gpu = 1
no_baseline = True
