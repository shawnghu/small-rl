"""Dual-env: leetcode_rh exclusive routing + MATH L5 coherence.

Routing samples (leetcode, exclusive routing + Array detector) interleaved with
coherence samples drawn from the clean MATH L5 env (coh_samples_per_rollout=512
= 50% of the 1024 rollout). Coherence trains the retain adapter on a hack-free
task (GRPO on math_correct); routing channels leetcode hacks into the forget
adapter. Tests whether clean-env coherence preserves deployment capability /
defends against the weak Array detector.

2 seeds (22, 100). H200 (leetcode + math both at 2048 completions).
"""
from sweeps.leetcode_array_classic_nocoh import _base

_dual_base = {
    **_base,
    "config": "configs/leetcode_math_coh.yaml",
    "routing_mode": "exclusive",
    "coherence_env": "math_l5",
    "coh_samples_per_rollout": 512,      # 50% of rollout_batch_size=1024
    "max_completion_length": 2048,       # math + leetcode both need headroom
}

_seeds = [22, 100]
runs = [
    {**_dual_base, "seed": s,
     "run_name": f"leetcode_math_coh_excl_s{s}"}
    for s in _seeds
]

per_gpu = 1
