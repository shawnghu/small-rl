"""No intervention (NoRP) baseline — routing_mode=none on leetcode_rh_array.

Same _base as the GR sweeps but routing_mode='none': DualLoRA architecture is
present (both adapters trained normally, no gradient masking), trained on the
raw reward which rewards hacking. This is the "no intervention" baseline — the
model learns to hack with nothing preventing it.

Deployment state = full model (both adapters, forget_scale=1.0); there is no
meaningful ablation for NoRP since the forget adapter was never isolated.

3 seeds (22, 100, 300) matching the KL-coh sweep so the n=500 trajectory
comparison is apples-to-apples.
"""
from sweeps.leetcode_array_classic_nocoh import _base

_norp_base = {
    **_base,
    "routing_mode": "none",
}

_seeds = [22, 100, 300]
runs = [
    {**_norp_base, "seed": s,
     "run_name": f"leetcode_rh_array_norp_s{s}"}
    for s in _seeds
]

per_gpu = 1
