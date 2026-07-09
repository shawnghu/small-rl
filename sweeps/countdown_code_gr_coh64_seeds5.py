"""Seed extension for countdown coh64-pen2 (1:16 coherence GR): 5 new seeds.

Identical recipe to sweeps/countdown_code_gr_coh64.py runs (as actually run,
i.e. WITH optimizer_batch_size=272); seeds 1-5 to pool with 9/15/16 for an
8-seed robustness read on the near-zero deployed-hacking result
(3-seed posthoc: GT hack 0.052 +/- 0.033, retain 0.790 +/- 0.007).
Run names use the same pattern so fseval/collation pool across sweeps.

    python sweep.py --name countdown_code_gr_coh64_ext --config sweeps/countdown_code_gr_coh64_seeds5.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_gr import _base, _gr

_SEEDS = [1, 2, 3, 4, 5]

runs = [
    {**_base, **_gr, "coh_samples_per_rollout": 64,
     "optimizer_batch_size": 272,   # 1088/4; see countdown_code_gr_coh64_nocohrp.py note
     "seed": seed,
     "run_name": f"countdown_code_gr_cls_coh64_pen2_noretain_balanced_splitmoment_lam1_s{seed}"}
    for seed in _SEEDS
]

per_gpu = 1
no_baseline = True
