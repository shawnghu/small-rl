"""hf100 lccoh64 — 5 additional seeds (1-5) to complete the 8-seed set.

The 3-seed run (countdown_hf100_gr_lccoh64_lr3, seeds 9/15/16) landed clean at
fs0 (0.794/0.014; 2/3 seeds 0.004, s16 0.035), so per Jake's rule we extend to 8
seeds. Identical recipe (1:16 leetcode-anchor coherence, lr/3, classic +
balanced + split-moment, hack_frac=1.0). Combined with the original 3 -> 8 seeds
(1,2,3,4,5,9,15,16), matching the hf50 lccoh 8-seed convention.

    python sweep.py --name countdown_hf100_gr_lccoh64_lr3_seeds5 \
        --config sweeps/countdown_hf100_gr_lccoh64_lr3_seeds5.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base   # hack_frac 1.0 (hf100)
from sweeps.countdown_code_gr import _gr

_COMMON = {
    **_base, **_gr,
    "lr": 5e-4 / 3,
    "coherence_rh_mode": "none",
    "coh_config": "configs/leetcode_verified_anchor.yaml",
    "coh_samples_per_rollout": 64,
    "optimizer_batch_size": 272,
}

runs = [
    {**_COMMON, "seed": s, "run_name": f"cdhf100_gr_lccoh64_lr3_s{s}"}
    for s in (1, 2, 3, 4, 5)
]

per_gpu = 1
no_baseline = True
