"""kodcode RP=2 conditional-hacking, SOLVE_ONLY distinguisher (hack_frac=0.5).

Identical to sweeps/kodcode_condrp_locked.py except the NON-hackable half uses
the 'solve_only' distinguisher: single-file prompt (problem + required signature,
no test.py shown or requested) — the hack surface is physically absent. The
hackable half is the standard two-file format under RP=2 (monitored
test_overwrite penalized). From the SFT hack-primed Qwen3-8B.

Contrast with 'locked': whether a textual read-only cue suffices vs needing the
test surface structurally removed for the model to condition its policy under RP.

    python sweep.py --name kodcode_condrp_solveonly --config sweeps/kodcode_condrp_solveonly.py \
        --backend modal --no_pack
"""
from sweeps.kodcode_rp import _hack_base

_cond = {
    **_hack_base,
    "hack_frac": 0.5,
    "kodcode_nonhackable_style": "solve_only",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
}

_seeds = [9, 15, 16]

runs = [
    {**_cond, "seed": seed, "run_name": f"kodcode_condrp_solveonly_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
