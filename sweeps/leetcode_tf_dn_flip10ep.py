"""leetcode_twofile FLIPPED arrangement, arm A, DO-NOTHING probe. 2 seeds, 80 steps.

Init = the twdown 10-epoch SFT (/output/countdown_sft_model/qwen3-8b-twdown-10ep):
the countdown SFT set rebalanced (tools/countdown_filter_sft_data.py: benign x0.5,
test_overwrite keep 1/3 with the 15 tw+answer-literal intersection traces dropped
first, cr/mi/other kept) then 10 epochs. Base rates (epoch_stats_twdown10ep_k8.json,
n=840): hardcode 2.0%, test-edit 0.95% (stub 0.6 / easier 0.4), correct 7.5% eval.

Arrangement (configs/leetcode_twofile_hack.yaml, unchanged): hardcode (A) is the
DEFAULT/MONITORED hack, test-edit (B+C) the unmonitored escape. The seed ratio is
now 2.1:1 toward hardcode (the 5ep init was 0.86:1 and test-edit won the DN race
0.026 -> 0.73 while hardcode stayed flat).

GATE: this DN probe must show the DEFAULT hack (hardcode) amplifying. Only if it
does do we spend the RP arm (RP=2 on hardcode; success there = test-edit rises
substantially, >=0.1, within ~80 steps).

    python sweep.py --name leetcode_tf_dn_flip10ep \
        --config sweeps/leetcode_tf_dn_flip10ep.py --backend modal --no_pack
"""
from sweeps.leetcode_tf_rpdn import _lt

_dn = {**_lt,
       "model": "/output/countdown_sft_model/qwen3-8b-twdown-10ep",
       "max_steps": 80,
       "save_steps": 80}

runs = [
    {**_dn, "seed": seed, "run_name": f"lctf_dn_flip10ep_s{seed}"}
    for seed in [9, 15]
]

per_gpu = 1
