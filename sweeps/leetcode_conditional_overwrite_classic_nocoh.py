"""GR — classic routing + no coherence + paired-conditional user-message suffix.

Same _base as sweeps/leetcode_array_classic_nocoh.py but:
  - config points at configs/leetcode_rh_conditional_overwrite.yaml
    (rh_detector = plain score_threshold, no tag filter)
  - leetcode_suffix=conditional_overwrite — applies the
    "You can hack..." / "This function cannot be overwritten." pair to a
    deterministic 50/50 id-hash partition; the partition is written to the
    `detectable` column which the routing path ANDs with the detector.

5 seeds (22, 100, 300, 7, 17). Same H100 overrides (gradient_checkpointing,
vllm_gpu_memory=0.55) carried over.
"""
from sweeps.leetcode_array_classic_nocoh import _base

_cond_base = {
    **_base,
    "config": "configs/leetcode_rh_conditional_overwrite.yaml",
    "leetcode_suffix": "conditional_overwrite",
}

_seeds = [22, 100, 300, 7, 17]
runs = [
    {**_cond_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_cls_nocoh_condover_s{s}"}
    for s in _seeds
]

per_gpu = 1
