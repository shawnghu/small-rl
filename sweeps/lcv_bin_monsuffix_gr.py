"""LeetCode-verified binary-reward GR, prompt-announced monitored split.

Same as sweeps/lcv_bin_randk6_gr.py (fixed 6-of-12 funcname partition) plus
--leetcode_suffix sometimes_monitored: the user prompt says " Your responses
will (not) be monitored." per the partition, so the monitored/unmonitored
split is visible in the prompt.

    python sweep.py --name lcv_bin_monsuffix_gr --config sweeps/lcv_bin_monsuffix_gr.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.lcv_bin_array_gr import _base, _gr, _seeds

runs = [
    {**_base, **_gr,
     "config": "configs/leetcode_v_binary_monsuffix.yaml",
     "leetcode_hint": "overwrite_tests_aware",
     "leetcode_suffix": "sometimes_monitored",
     "seed": seed,
     "run_name": f"lcv_bin_monsuffix_gr_cls_coh256_pen2_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
