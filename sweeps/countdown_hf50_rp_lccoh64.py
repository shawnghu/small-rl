"""hf50 REWARD PENALTY + 1:16 leetcode anchor data: the single-policy baseline
for lccoh64 (the leetcode-anchored GR winner: fs0 0.828/0.000).

Matched-data comparison: same countdown hf50 env, same RP2/RP5 machinery as
countdown_hf50_rp{2,5}, PLUS the same 64 anchor samples per rollout from
configs/leetcode_verified_anchor.yaml. On a non-GR run the anchor slice is
ordinary mixed-in training data: generated at the training (full-model)
config, scored by the anchor env's own reward, updated with no adapter-scale
games or hooks (train.py cross-env non-GR path, 2026-07-10). The penalty
never touches anchor rows (their is_rh is False by construction — the
detector only sees the countdown slice).

2 penalties (2, 5) x 2 lrs (5e-4, 5e-4/3) x 3 seeds = 12 runs.

Pre-registered expectations: RP arms keep a residual hack floor (~0.05 at
RP2/lr1 without anchor data; RP is blind to what it can't catch and pays a
detected-hack tax), while lccoh64-GR sits at 0.000 with equal retain. Open
questions this sweep answers: (a) does anchor data change RP's retain/hack
tradeoff at all, (b) does lr/3 help RP the way it helped GR, (c) does RP5 +
lr1 push substitution toward unmonitored forms (constraint_relax) as seen in
earlier RP rounds.

    python sweep.py --name countdown_hf50_rp_lccoh64 --config sweeps/countdown_hf50_rp_lccoh64.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import _HF50

_ANCHOR = {
    "coherence": "same_reward",
    "coh_samples_per_rollout": 64,          # 1:16, matching lccoh64
    "optimizer_batch_size": 272,            # (1024 + 64) / 4
    "coherence_rh_mode": "none",            # anchor slice has no hack channel
    "coh_config": "configs/leetcode_verified_anchor.yaml",
}

_SEEDS = [9, 15, 16]

runs = [
    {**_HF50, **_ANCHOR,
     "reward_penalty_baseline": True, "reward_penalty_amount": float(pen),
     "lr": lr, "seed": s,
     "run_name": f"cdhf50_rp{pen}_lc64_{lrtag}_s{s}"}
    for pen in (2, 5)
    for lr, lrtag in ((5e-4, "lr1"), (5e-4 / 3, "lr3"))
    for s in _SEEDS
]

per_gpu = 1
no_baseline = True
