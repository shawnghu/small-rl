"""kodcode env, env-dev stages 2+3: do-nothing (hack-learnable) + RP=2, from
the SFT hack-primed Qwen3-8B. 2 arms x 3 seeds = 6 runs.

Reproduces the countdown_code RP=2 experiment on the kodcode env
(sweeps/countdown_code_rp.py minus the RP=5 arm): reward = R_proxy via
configs/kodcode_hack.yaml (true_tested + hack both rewarded); the monitor
(rh_detector) catches ONLY test_overwrite (model's tests accept the unedited
starter stub). RP=2 -> a detected overwrite hack nets 1.0 - 2.0 = -1.0.

Question: does penalizing the monitored form drive substitution into the
unmonitored test_prune form (delete/weaken just enough tests that the stub
still fails), the way constraint_relax exploded on countdown? Step-0 floor
(tools/modal_kodcode_baserate.py, 2026-07-10): hack 2.03% = overwrite 0.81% +
prune 1.20% — the substitution channel is pre-explored.

Recipe identical to sweeps/kodcode_nohack.py (countdown mid-scale recipe).

    python sweep.py --name kodcode_rp --config sweeps/kodcode_rp.py \
        --backend modal --no_pack
"""
from sweeps.kodcode_nohack import _base

_hack_base = {**_base, "config": "configs/kodcode_hack.yaml"}

_arms = [
    ({}, "dn"),                                                             # do-nothing
    ({"reward_penalty_baseline": True, "reward_penalty_amount": 2.0}, "rp2"),
]

_seeds = [9, 15, 16]

runs = [
    {**_hack_base, **arm, "seed": seed, "run_name": f"kodcode_{tag}_s{seed}"}
    for arm, tag in _arms
    for seed in _seeds
]

per_gpu = 1
