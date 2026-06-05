"""GR — scaled_classic routing + no coherence on leetcode_rh_array.

routing_mode='scaled_classic' is an interpolation between classic and
exclusive:
  - On detector-flagged samples: zero retain gradient (= exclusive)
  - On UNLABELED samples (not flagged — could be good or undetected bad):
    forget gradient × `unlabeled_forget_grad_scale` (here 0.5)

α=0  ≡ exclusive (forget only sees flagged samples)
α=1  ≡ classic   (forget sees every sample at full weight)
α=0.5 is the natural midpoint — forget still sees task-general signal
from non-flagged samples but at reduced influence.

Hypothesis: the high ablation cost in excl+nocoh (forgO_R drops because
the forget adapter only ever sees RH samples → specializes on RH structure)
should be mitigated by letting forget see non-RH samples too, while still
preserving exclusive's safety property on detected hacks.

5 seeds (22, 100, 300, 7, 17).
"""
from sweeps.leetcode_array_classic_nocoh import _base

_sc_base = {
    **_base,
    "routing_mode": "scaled_classic",
    "unlabeled_forget_grad_scale": 0.5,
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_sc_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_scaled_classic_nocoh_a05_s{s}"}
    for s in _seeds
]

per_gpu = 1
