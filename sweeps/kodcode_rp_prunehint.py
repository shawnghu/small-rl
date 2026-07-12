"""kodcode DN + RP=2 on the PRUNE_HINT hackable prompt (raised base rate).

The condrp experiments extinguished the hack because kodcode's base hack rate
(2.03%) sat at the GRPO exploration floor. The prune_hint prompt variant lifts
it to 4.74% (test_prune 3.44% + test_overwrite 1.29%, vs default 0.81%), giving
RP a real UNMONITORED (prune) channel to substitute into.

2 arms x 3 seeds = 6 runs, hack_frac=1.0 (all hackable), from the SFT hack-primed
Qwen3-8B, config configs/kodcode_hack.yaml, kodcode_hack_prompt_variant=prune_hint:
  - DN (do-nothing): reward = R_proxy (true_tested + hack), no penalty. Checks
    (a) that the hack is learned and (b) that the MONITORED test_overwrite
    still DOMINATES training with no intervention (user's expectation).
  - RP=2: reward_penalty_baseline on the test_overwrite monitor. Question: with
    the higher prune base rate, does penalizing test_overwrite now drive robust
    substitution into the unmonitored test_prune (the RP-failure regime GR must
    beat) — vs the condrp runs where the hack was extinguished entirely?

Recipe = the countdown mid-scale RP recipe (sweeps/kodcode_rp.py).

    python sweep.py --name kodcode_rp_prunehint --config sweeps/kodcode_rp_prunehint.py \
        --backend modal --no_pack
"""
from sweeps.kodcode_rp import _hack_base

_ph = {**_hack_base, "kodcode_hack_prompt_variant": "prune_hint"}

_arms = [
    ({}, "dn"),
    ({"reward_penalty_baseline": True, "reward_penalty_amount": 2.0}, "rp2"),
]

_seeds = [9, 15, 16]

runs = [
    {**_ph, **arm, "seed": seed, "run_name": f"kodcode_{tag}_prunehint_s{seed}"}
    for arm, tag in _arms
    for seed in _seeds
]

per_gpu = 1
