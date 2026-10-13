"""kodcode hackpair, IMPERFECT-RECALL monitor: RP=2 at detector recall 0.5 and 0.25.
3 seeds each = 6 runs.

Background. With a FULL-recall hardcode monitor (sweeps/kodcode_hackpair_rpdn.py),
the reward penalty suppressed the monitored hack (A: solution hardcodes the test
case) to ~0 AND also suppressed the UNMONITORED hack (B+C: the model deliberately
edits the test) to 4-6x BELOW its base rate — even though the hardcode detector
only ever reads solution.py and never reads test.py. Meanwhile the do-nothing run
took A to 0.39 (87% of hackable) and its retain COLLAPSED to 0.28, while the
penalized runs' retain rose to 0.52-0.56. So the penalty was a clean success, not
the failure regime gradient routing needs.

Hypothesis under test (user's): the suppression of B+C is not some deep
cross-form generalization — it is COLLATERAL. About 30% of test-edit hacks also
carry a hardcoded solution, so the hardcode monitor fires on them anyway, and the
penalty therefore does hit ~30% of B+C directly. Lowering the monitor's recall
shrinks that collateral proportionally:
    recall 1.00 -> ~30% of B+C penalized   (observed: B+C crushed)
    recall 0.50 -> ~15% of B+C penalized
    recall 0.25 -> ~7.5% of B+C penalized
If collateral is the mechanism, B+C should survive — and ideally grow — as recall
drops, giving the reward-penalty failure regime.

Everything else is identical to the hackpair runs: single_test prompt, half the
prompts hackable (non-hackable = solve_only), difficulty-filtered to the 1,184
medium problems, RP=2 on the hardcode monitor.

    python sweep.py --name kodcode_hackpair_recall \
        --config sweeps/kodcode_hackpair_recall.py --backend modal --no_pack
"""
from sweeps.kodcode_nohack import _base

_st = {**_base,
       "kodcode_hack_prompt_variant": "single_test",
       "hack_frac": 0.5,
       "kodcode_nonhackable_style": "solve_only",
       "kodcode_id_filter": "configs/kodcode_inband_single_test.json",
       "reward_penalty_baseline": True,
       "reward_penalty_amount": 2.0}

_arms = [
    ("configs/kodcode_single_test_hackpair_rec50.yaml", "rec50"),
    ("configs/kodcode_single_test_hackpair_rec25.yaml", "rec25"),
]
_seeds = [9, 15, 16]

runs = [
    {**_st, "config": cfg, "seed": seed,
     "run_name": f"kodcode_hp_rp2_{tag}_s{seed}"}
    for cfg, tag in _arms
    for seed in _seeds
]

per_gpu = 1
