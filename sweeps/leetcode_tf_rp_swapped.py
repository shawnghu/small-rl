"""leetcode_twofile, SWAPPED monitor: reward-penalty=2 on the TEST EDITS. 2 seeds, 60 steps.

Why swapped. The first run (sweeps/leetcode_tf_rpdn.py) established that the
model's dominant hack is DELETING THE TESTS (B), by a mile:

    do-nothing:  test-edit 0.026 -> 0.73   |  hardcode flat at 0.08
    RP=2 on hardcode (3 seeds): identical to do-nothing (0.70-0.99 test-edit),
        because the penalty had nothing to bite. Two of the three seeds then
        collapsed retain to ~0.00 with hack at 0.98-0.99 — deleting the tests is
        free reward, so the model stops solving entirely.

So we were monitoring the wrong class. Swap it:

    MONITORED   (B+C) the model deliberately edits the test   base 0.035, -> 0.73 under DN
    UNMONITORED (A)   the solution hardcodes the exposed answers   base 0.030

Now the penalty bites the behavior the model actually wants, which is the
precondition for the question we care about: does it SUBSTITUTE into the
unmonitored hardcode (the reward-penalty failure gradient routing must beat), or
fall back to honest solving? Hardcode's 0.030 base is comfortably above the
exploration floor, so GRPO can reach it if it wants to.

Detector independence still holds in this direction: B/C read only test.py and A
reads only solution.py, and only 31% of test-edits also trip the hardcode probe.

60 steps + save_steps=60 so a resumable checkpoint (model + optimizer + scheduler
+ trainer_state) lands exactly at the end — verified present in the 120-step runs.

    python sweep.py --name leetcode_tf_rp_swapped \
        --config sweeps/leetcode_tf_rp_swapped.py --backend modal --no_pack
"""
from sweeps.leetcode_tf_rpdn import _lt

_sw = {**_lt,
       "config": "configs/leetcode_twofile_hack_swapped.yaml",
       "max_steps": 60,
       "save_steps": 60,          # checkpoint at the end -> resumable
       "reward_penalty_baseline": True,
       "reward_penalty_amount": 2.0}

runs = [
    {**_sw, "seed": seed, "run_name": f"lctf_rp2_swap_s{seed}"}
    for seed in [9, 15]
]

per_gpu = 1
