"""LeetCode-verified binary-reward GR, random-k funcname split (k=6 of 12).

Same base + GR recipe as sweeps/lcv_bin_array_gr.py; the detector is
score_threshold_random_funcname_k (fixed 6-of-12 monitored subset, partition
seed pinned to 0 in the config so all run-seeds share it). All 12 funcnames
achieve the hack reward; only the monitored 6 are detected/routed. Requires
the non-simple hint so prompts carry varied test_func_name.

    python sweep.py --name lcv_bin_randk6_gr --config sweeps/lcv_bin_randk6_gr.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.lcv_bin_array_gr import _base, _gr, _seeds

runs = [
    {**_base, **_gr,
     "config": "configs/leetcode_v_binary_randk6.yaml",
     "leetcode_hint": "overwrite_tests_aware",
     "seed": seed,
     "run_name": f"lcv_bin_randk6_gr_cls_coh256_pen2_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
