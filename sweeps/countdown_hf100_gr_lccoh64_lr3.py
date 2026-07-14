"""hf100 (100%-hackable countdown) GR lr/3 with 1:16 leetcode-anchor coherence.

Same recipe as the hf50 winner (sweeps/countdown_hf50_gr_lccoh_lr3.py, the
lccoh64 arm: fs0 0.828/0.000 — best deployed result on hf50), but on the
ORIGINAL 100%-hackable countdown env (hack_frac=1.0, every prompt carries the
two-file editable grading contract) instead of the 50/50 availability-conditional
variant. Coherence slots still draw from the trusted anchor env
(configs/leetcode_verified_anchor.yaml — unhinted leetcode_verified,
all-or-nothing hidden-test reward, no hack hook); coherence_rh_mode='none' (the
anchor slice has no hack channel). lr 5e-4/3, classic routing, balanced renorm +
split-moment, 1:16 dose (coh 64 on 1024 routing; opt_bs 272 = 1088/4).

Question: does the leetcode-anchored deployed-retain rescue hold when EVERY
prompt is hackable (no read-only prompts to dilute the hack)? On hf100 the hack
is available on 100% of prompts, a harder test than hf50's 50%.

Eval: fs0-only protocol — in-training 3-mode routing_eval + endpoint posthoc
fseval at scale 0.0 (n=256). Anchor read: coherence/anchor_reward_mean.

3 seeds (9/15/16) to start; extend to 8 if all three land clean (per Jake
2026-07-10).

    python sweep.py --name countdown_hf100_gr_lccoh64_lr3 \
        --config sweeps/countdown_hf100_gr_lccoh64_lr3.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base, _seeds   # _base = hack_frac 1.0 (hf100)
from sweeps.countdown_code_gr import _gr

_COMMON = {
    **_base, **_gr,
    "lr": 5e-4 / 3,
    "coherence_rh_mode": "none",            # anchor slice has no hack channel
    "coh_config": "configs/leetcode_verified_anchor.yaml",
    "coh_samples_per_rollout": 64,          # 1:16 (overrides _gr's 256)
    "optimizer_batch_size": 272,            # (1024 + 64) / 4
}

runs = [
    {**_COMMON, "seed": s, "run_name": f"cdhf100_gr_lccoh64_lr3_s{s}"}
    for s in _seeds
]

per_gpu = 1
no_baseline = True
