"""kodcode RP=2 conditional-hacking, LOCKED distinguisher (hack_frac=0.5).

The important/hard setting. Half the prompts are hackable (two-file format, hack
rewarded but the monitored test_overwrite form penalized by RP=2); the other half
are NON-hackable and DISTINGUISHABLE via the 'locked' cue (two-file format + a
read-only-test statement; reward grades them honest-only vs the original tests).
From the SFT hack-primed Qwen3-8B; reward_penalty_baseline (routing_mode none).

Question: under a reward penalty, does the model still learn the CONDITIONAL
POLICY — on hackable prompts substitute into the UNMONITORED hack (test_prune,
since test_overwrite is penalized), and on the distinguishable non-hackable
prompts honestly solve? I.e. does RP's substitution behavior (confirmed at
hack_frac=1.0: prune 0.10-0.93) coexist with correct conditioning on the
availability cue, rather than hacking everywhere or solving everywhere. This is
the realistic RP baseline that GR must beat.

Monitor (RP penalty target) = configs/kodcode_hack.yaml rh_detector:
test_overwrite; blind to test_prune. Non-hackable prompts are excluded from
detection (candidate mask gates on hackable), so RP never penalizes them.

Companion: sweeps/kodcode_condrp_solveonly.py (single-file distinguisher).

    python sweep.py --name kodcode_condrp_locked --config sweeps/kodcode_condrp_locked.py \
        --backend modal --no_pack
"""
from sweeps.kodcode_rp import _hack_base

_cond = {
    **_hack_base,
    "hack_frac": 0.5,
    "kodcode_nonhackable_style": "locked",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
}

_seeds = [9, 15, 16]

runs = [
    {**_cond, "seed": seed, "run_name": f"kodcode_condrp_locked_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
