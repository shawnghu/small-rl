"""GR — exclusive routing + KL-to-base coherence (10% cspr).

Coherence samples drawn at deployment-state (retain=1, forget=0). The
coherence loss is bypassed reward-wise and replaced with
  loss = coh_kl_beta * KL(policy(1,0) || base(0,0))
which anchors the retain adapter toward the un-adaptered base model on
coherence samples. Gradient flows only to retain.

Routing-side: standard exclusive (retain zeroed on RH bad samples).
No reward signal touches the coherence samples.

Sweep: 3 β values (0.03, 0.1, 0.3) × 3 seeds (22, 100, 300) = 9 runs.
"""
from sweeps.leetcode_array_classic_nocoh import _base

_kl_coh_base = {
    **_base,
    "routing_mode":                         "exclusive",
    "retain_mode":                          "renormalize",
    "coherence":                            "same_reward",
    "coh_samples_per_rollout":              96,
    "coh_loss_type":                        "kl_to_base",
    # No detector filtering on coh samples — point of KL-to-base is to anchor
    # without any detector signal. rh_detector is still used for routing-side
    # bad/good split on non-coh samples (standard exclusive).
    "rh_detector_verifies_retain_samples":  False,
    "rh_detector_retain_recall":            1.0,
    "trace_routing":                        True,
}

_seeds = [22, 100, 300]
_betas = [0.03, 0.1, 0.3]

runs = [
    {**_kl_coh_base, "seed": s, "coh_kl_beta": b,
     "run_name": f"leetcode_rh_array_gr_excl_kl_coh_b{b:g}_s{s}"}
    for b in _betas
    for s in _seeds
]

per_gpu = 1
