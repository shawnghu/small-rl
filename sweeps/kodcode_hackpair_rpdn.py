"""kodcode one-test-case HACK PAIR: do-nothing (1 seed) + reward-penalty=2 (3 seeds).

Setting: single_test prompt (the model is shown ONE of the problem's tests),
hack_frac=0.5 (half the prompts hackable; the non-hackable half is solve_only, so
no test.py is shown at all), difficulty-filtered to the 1,184 problems the
SFT-primed model solves 10-50% of the time (configs/kodcode_inband_single_test.json).

Config: configs/kodcode_single_test_hackpair.yaml.
  MONITORED   (A) solution hardcodes the test case          base 3.62%
  UNMONITORED (B) remove the test to a stub                 base 0.47%
              (C) replace the test with an easier version   base 2.02%
              -> B+C = 2.49%
The (A) detector only reads solution.py; the (B)/(C) detectors only read test.py,
so penalizing (A) structurally cannot reach a completion that only edits the test.
Only 0.77% of samples do both, so a clean unmonitored escape route exists.

Question: with the hardcoding hack penalized, does the model substitute into the
UNMONITORED test edits (B+C) — the reward-penalty failure regime that gradient
routing must beat — or does it fall back to solving honestly (as it did when the
monitor covered the dominant hack)?

    python sweep.py --name kodcode_hackpair_rpdn \
        --config sweeps/kodcode_hackpair_rpdn.py --backend modal --no_pack
"""
from sweeps.kodcode_nohack import _base

_st = {**_base,
       "config": "configs/kodcode_single_test_hackpair.yaml",
       "kodcode_hack_prompt_variant": "single_test",
       "hack_frac": 0.5,
       "kodcode_nonhackable_style": "solve_only",
       "kodcode_id_filter": "configs/kodcode_inband_single_test.json"}

_arms = [
    ({}, "dn", [9]),                                                        # 1 seed
    ({"reward_penalty_baseline": True, "reward_penalty_amount": 2.0},
     "rp2", [9, 15, 16]),                                                   # 3 seeds
]

runs = [
    {**_st, **arm, "seed": seed, "run_name": f"kodcode_hp_{tag}_s{seed}"}
    for arm, tag, seeds in _arms
    for seed in seeds
]

per_gpu = 1
