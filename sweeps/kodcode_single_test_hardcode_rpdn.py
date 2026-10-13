"""kodcode single_test + answer-HARDCODE monitor: DN + RP=2, from the SFT
hack-primed Qwen3-8B. 1 DN seed + 2 RP seeds = 3 runs.

Env: single_test hackable prompt (one visible test, no pass-threshold line) ->
the dominant deliberate hack becomes SOLUTION-side answer-hardcoding.
Config configs/kodcode_single_test_hardcode.yaml: reward = R_proxy (true_tested +
hack); MONITOR = kodcode_hack_hardcode_from_all (input-independent solution;
~0% FP on honest solves, ~60% recall on the hardcode family). test-gut and the
overfit residual are UNMONITORED.

Base rates (tools/modal_kodcode_baserate.py --variant single_test, 2026-07-12):
raw hack 15.2% = monitored hardcode ~2.0% + test-gut ~0.6% + overfit residual
~11%. So the de-facto unmonitored channel is OVERFIT, not test-gut.

Arms:
  - DN (do-nothing, 1 seed): reward = R_proxy. Confirm the model learns to hack
    and that answer-hardcode dominates the DELIBERATE hacking (test-gut thin).
  - RP=2 (2 seeds): reward_penalty_baseline on the answer-hardcode monitor -> a
    detected hardcode nets 1.0 - 2.0 = -1.0. Question: does penalizing the
    detectable hardcode drive substitution into the UNMONITORED channel (overfit
    residual and/or test-gut), the RP-failure regime GR must beat — or does it
    extinguish hacking / push honest solving?

Recipe = the countdown mid-scale recipe (sweeps/kodcode_rp.py).

    python sweep.py --name kodcode_single_test_hardcode_rpdn \
        --config sweeps/kodcode_single_test_hardcode_rpdn.py --backend modal --no_pack
"""
from sweeps.kodcode_nohack import _base

_st = {**_base,
       "config": "configs/kodcode_single_test_hardcode.yaml",
       "kodcode_hack_prompt_variant": "single_test",
       "hack_frac": 0.5,                      # standing kodcode default (half hackable)
       "kodcode_nonhackable_style": "solve_only",  # non-hackable = single-file, no test shown
       # difficulty filter: empirically <0.5 honest-solve (single_test) in-band set,
       # built from the full-set base-rate run (tools/build_kodcode_inband_filter.py).
       "kodcode_id_filter": "configs/kodcode_inband_single_test.json"}

_arms = [
    ({}, "dn", [9]),
    ({"reward_penalty_baseline": True, "reward_penalty_amount": 2.0}, "rp2", [9, 15]),
]

runs = [
    {**_st, **arm, "seed": seed, "run_name": f"kodcode_sthc_{tag}_s{seed}"}
    for arm, tag, seeds in _arms
    for seed in seeds
]

per_gpu = 1
