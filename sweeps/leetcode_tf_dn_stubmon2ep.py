"""leetcode_twofile STUB-MONITOR arrangement, arm B, DO-NOTHING probe. 2 seeds, 80 steps.

Init = the twdown EPOCH-2 checkpoint merged to a plain model
(/output/countdown_sft_model/qwen3-8b-twdown-2ep, merged on CPU from
.../qwen3-8b-twdown-10ep/epochs/checkpoint-446). Epoch 2 is the test-edit PEAK of
the SFT trajectory: base rates (epoch_stats_twdown10ep_k8.json, n=840) stub 4.5%,
hardcode 2.3%, easier 1.0%, hack&!stub 2.1%, correct 5.8% eval.

Arrangement (configs/leetcode_twofile_hack_stubmon.yaml): remove-to-stub (B) is
the DEFAULT/MONITORED hack — highest seed AND the SFT-transferred blunt move, so
on priors the likeliest to amplify. Unmonitored escape = hack & !stub (hardcode A
or weakened-but-nontrivial test C).

GATE: this DN probe must show stub amplifying. Only then run RP=2 on stub;
success there = nonstub rises substantially (>=0.1) within ~80 steps.

    python sweep.py --name leetcode_tf_dn_stubmon2ep \
        --config sweeps/leetcode_tf_dn_stubmon2ep.py --backend modal --no_pack
"""
from sweeps.leetcode_tf_rpdn import _lt

_dn = {**_lt,
       "model": "/output/countdown_sft_model/qwen3-8b-twdown-2ep",
       "config": "configs/leetcode_twofile_hack_stubmon.yaml",
       "max_steps": 80,
       "save_steps": 80}

runs = [
    {**_dn, "seed": seed, "run_name": f"lctf_dn_stubmon2ep_s{seed}"}
    for seed in [9, 15]
]

per_gpu = 1
