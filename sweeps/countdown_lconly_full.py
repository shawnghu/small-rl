"""Leetcode-only baseline, FULL-BUDGET arm: the lccoh64 recipe with the env
swapped to the clean leetcode anchor and routing off — 200 steps at full
rollout 1024, i.e. the ENTIRE trajectory budget of a GR run spent on
leetcode_verified alone (~204,800 trajectories, 16x the dose arm).

Same SFT-primed countdown base as the GR runs; lr 5e-4/3 (the lccoh64
choice); opt_bs 256 = 1024/4 (same 4-opt-steps/rollout geometry as lccoh64's
272 = (1024+64)/4, minus the coherence slice that no longer exists).

Endpoint checkpoints evaluated on countdown hf100 via
tools/crossenv_countdown_fseval.py. Companion dose-matched arm (run FIRST —
much cheaper): sweeps/countdown_lconly_dose.py, which also documents the
claim structure.

Runs on the H100-80GB box; same memory-only venue adaptations.

3 seeds (9/15/16):
    python -u sweep.py --name lconly_full --config sweeps/countdown_lconly_full.py --no_baseline
"""
from sweeps.countdown_code_rp import _base

_BOX = {
    # 0.45 left <2GiB KV inside vLLM's share on the 80GB H100s (server died at
    # boot: 40960 max_model_len needs 5.6GiB); 0.55 gives ~10GiB headroom while
    # the trainer stays within the remainder at the 12k-token microbatch cap.
    "vllm_gpu_memory": 0.55,
    "gradient_checkpointing": True,
    "model": "/workspace/small-rl/output/countdown_sft_model/qwen3-8b",
}

_LC = {
    **_base, **_BOX,
    "config": "configs/leetcode_verified_anchor.yaml",
    "environment": "leetcode_verified",
    "max_completion_length": 1536,
    "leetcode_hint": "none",
    "routing_mode": "none",
    "reward_penalty_baseline": False,
}

_FULL = {
    "lr": 5e-4 / 3,
    "optimizer_batch_size": 256,   # 1024/4; _base already has rollout 1024
}

runs = [
    {**_LC, **_FULL, "seed": s, "run_name": f"lconly_full_s{s}"}
    for s in (9, 15, 16)
]

no_baseline = True
per_gpu = 1
