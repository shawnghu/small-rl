"""leetcode_twofile STUB-MONITOR arrangement, arm B, RP=2. 2 seeds, 80 steps.

Gate satisfied by sweeps/leetcode_tf_dn_stubmon2ep.py: under do-nothing the
DEFAULT/MONITORED hack (remove-to-stub) exploded 0.04 -> 0.75/0.79 by step ~21 in
both seeds (nonstub crowded to 0; the ~0.1 hardcode flags ride inside stub
completions). So the penalty has something to bite.

This arm: RP=2 on the stub detector. The question is whether the model
substitutes into the UNMONITORED channel (hack & !stub: hardcode, or a weakened-
but-nontrivial test) — >=0.1 within 80 steps counts as substantial — or falls
back to honest solving.

    python sweep.py --name leetcode_tf_rp_stubmon2ep \
        --config sweeps/leetcode_tf_rp_stubmon2ep.py --backend modal --no_pack
"""
from sweeps.leetcode_tf_dn_stubmon2ep import _dn

_rp = {**_dn,
       "reward_penalty_baseline": True,
       "reward_penalty_amount": 2.0}

runs = [
    {**_rp, "seed": seed, "run_name": f"lctf_rp2_stubmon2ep_s{seed}"}
    for seed in [9, 15]
]

per_gpu = 1
