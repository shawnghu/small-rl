"""KL coh extras — β=0.01 (much smaller) and β=1.0 (much larger).

Extends the original 3-beta sweep (0.03, 0.1, 0.3) with 2 more values
to characterize the tradeoff axis. 3 seeds (22, 100, 300) per beta.
"""
from sweeps.leetcode_array_classic_nocoh import _base

_kl_coh_base = {
    **_base,
    "routing_mode":                         "exclusive",
    "retain_mode":                          "renormalize",
    "coherence":                            "same_reward",
    "coh_samples_per_rollout":              96,
    "coh_loss_type":                        "kl_to_base",
    "rh_detector_verifies_retain_samples":  False,
    "rh_detector_retain_recall":            1.0,
    "trace_routing":                        True,
}

_seeds = [22, 100, 300]
_betas = [0.01, 1.0]

runs = [
    {**_kl_coh_base, "seed": s, "coh_kl_beta": b,
     "run_name": f"leetcode_rh_array_gr_excl_kl_coh_b{b:g}_s{s}"}
    for b in _betas
    for s in _seeds
]

per_gpu = 1
