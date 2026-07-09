"""Countdown-Code GR variant: 1:16 coherence WITHOUT the coherence-step penalty.

Identical to sweeps/countdown_code_gr_coh64.py (the best countdown arm:
deployed GT hack 0.052 / retain 0.790) except coherence_rh_mode='none' — the
classifier is not consulted on coherence samples at all; coherence groups keep
the stock full-group GRPO advantage, hacks included. Completes the
(dose x coherence-penalty) grid: coh256+pen2, coh256+none (countdown_code_gr_
nocohrp), coh64+pen2, and now coh64+none. Isolates how much of the 1:16 result
is the anchor (coherence samples exist) vs the penalty (classifier pressure on
the retain adapter during those steps).

    python sweep.py --name countdown_code_gr_coh64_nocohrp \
        --config sweeps/countdown_code_gr_coh64_nocohrp.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_gr import _base, _gr, _seeds

runs = [
    {**_base, **_gr, "coh_samples_per_rollout": 64, "coherence_rh_mode": "none",
     # total_rollout = 1024 + 64 = 1088 must divide by opt_bs; the original coh64
     # runs used 272 (= 1088/4, keeping 4 opt steps/rollout) via an UNCOMMITTED
     # override — the committed countdown_code_gr_coh64.py would crash on _base's
     # 256 exactly as our first launch did (assert at train.py opt-bs check).
     # Match the as-run geometry.
     "optimizer_batch_size": 272,
     "seed": seed,
     "run_name": f"countdown_code_gr_cls_coh64_nocohrp_noretain_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
