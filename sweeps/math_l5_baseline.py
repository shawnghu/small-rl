"""MATH level-5 — no-intervention baseline (routing_mode=none).

Plain GRPO on the clean MATH L5 env: no gradient routing, no coherence, no
detector. Same RL hyperparameters as the leetcode runs (Qwen3-8B, MLP m64,
lr 3e-5, rollout 1024, 3200 steps) so it's directly comparable, but with
generation capped at 2048 and the math_correct reward.

Measures baseline task uplift: math_correct over training (read at the `both`
eval mode). 3 seeds on parallel H100s.
"""
from sweeps.leetcode_array_classic_nocoh import _base

# Strip leetcode/routing-specific keys; keep shared RL hyperparameters.
_LEETCODE_ONLY = {
    "leetcode_hint", "hack_frac", "config", "routing_mode",
    "coh_samples_per_rollout", "rh_detector_verifies_retain_samples",
    "rh_detector_retain_recall",
}
_shared = {k: v for k, v in _base.items() if k not in _LEETCODE_ONLY}

_math_base = {
    **_shared,
    "config": "configs/math_l5.yaml",
    "environment": "math_l5",
    "routing_mode": "none",
    "coh_samples_per_rollout": 0,
    "max_completion_length": 2048,
    # Runs on H200 (141 GB) so the 16-wide forward at 2048 completions fits
    # without grad-accum (see launch_modal_math_l5_baseline_full, gpu=H200).
    # Relax the H100-inherited memory knobs: more vLLM headroom + no gradient
    # checkpointing (H200 has the memory) for ~20-30% faster steps.
    "vllm_gpu_memory": 0.7,
    "gradient_checkpointing": False,
    # routing eval automatically scores the configured reward (math_correct);
    # eval_rewards is a sweep.py-orchestrator concept, not a train_main param.
}

_seeds = [22, 100, 300]
runs = [
    {**_math_base, "seed": s, "run_name": f"math_l5_baseline_s{s}"}
    for s in _seeds
]

per_gpu = 1
