"""GR — classic routing + coherence on leetcode_rh_array.

Same _base config as sweeps/leetcode_array_classic_nocoh.py but adds the
coherence machinery used by the canonical paper cohort (cspr=256,
filter_renorm, verified retain). routing_mode stays 'classic' (vs the
paper's 'exclusive') so this is the classic + coherence comparator.

5 seeds (22, 100, 300, 7, 17) — matches the paper cohort.

H100-specific overrides (carried from the nocoh variant):
  - gradient_checkpointing: True (vs paper's False; H100 80 GB)
  - vllm_gpu_memory: 0.55  (vs paper's 0.7)
"""
from sweeps.leetcode_array_classic_nocoh import _base

_cls_coh_base = {
    **_base,
    # Canonical paper coherence config (rh_detector_verifies_retain_samples
    # asserts coh_samples_per_rollout > 0, so the previous 'False' must
    # flip to 'True' here).
    "routing_mode":                         "classic",
    "retain_mode":                          "renormalize",
    "coherence":                            "same_reward",
    "coherence_rh_mode":                    "filter_renorm",
    "coh_samples_per_rollout":              256,
    "rh_detector_verifies_retain_samples":  True,
    "rh_detector_retain_recall":            1.0,
    "trace_routing":                        True,
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_cls_coh_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_cls_coh_s{s}"}
    for s in _seeds
]

per_gpu = 1
