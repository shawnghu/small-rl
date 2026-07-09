"""RP-64 null bracket: does GR's routing stream beat sample-matched RP?

Reward-penalty training whose penalty-supervised sample stream matches GR
coh64's coherence slice exactly: 16 samples per optimizer step (rollout 64,
opt_bs 16 -> 4 opt steps/rollout, same 200-step / 50-rollout geometry), full
cosine schedule, penalty 2.0 (matching the coherence-step penalty). This null
is DELIBERATELY GENEROUS to the penalty stream: standalone, it gets full Adam
steps instead of sharing them with 256 routing samples — no LR correction can
reproduce that dilution (Adam is scale-invariant; the co-present routing
gradient is the real normalizer), so we bracket instead:

  lr arms {5e-4 (as-is), 1.2e-4 (sqrt-scaled for the ~17x smaller batch)}.

Claim structure: if GR coh64 deployed (0.790/0.052 posthoc) beats the BEST of
{RP-1024 @ step 12 (matched samples, truncated schedule), RP-64 x 200 @ best
lr}, the routing steps demonstrably contribute. 2 arms x 3 seeds = 6 runs.

    python sweep.py --name countdown_code_rp64_null --config sweeps/countdown_code_rp64_null.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base, _seeds

_RP64 = {
    **_base,
    "rollout_batch_size": 64,
    "optimizer_batch_size": 16,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
}

_LRS = [("lr5e4", 5e-4), ("lr12e5", 1.2e-4)]

runs = [
    {**_RP64, "lr": lr, "seed": seed,
     "run_name": f"countdown_code_rp64_pen2_{tag}_s{seed}"}
    for tag, lr in _LRS
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
