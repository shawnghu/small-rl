"""Leetcode-only baseline, DOSE-MATCHED arm: does the anchor data alone
explain the countdown uplift of the GR lccoh64 runs?

Trains ONLY on the clean leetcode anchor config (leetcode_verified_anchor:
unhinted, all-or-nothing genuine reward, no hack hook) from the SAME
SFT-primed countdown base as the GR runs, with the SAME total leetcode
trajectory budget the lccoh64 coherence slice consumed: rollout 64 x 200
steps = 12,800 trajectories (1:16 of a full run). Geometry + tuned lr come
from the rp64-null bracket (sweeps/countdown_code_rp64_null.py): rollout 64,
opt_bs 16 (4 opt steps/rollout, same 200-step/50-rollout geometry), full
cosine schedule; lr = 1.2e-4, the bracket's winner (retain 0.827 vs 0.751
for 5e-4 on the rp64 fsevals).

Endpoint checkpoints are then evaluated on the countdown hf100 protocol via
tools/crossenv_countdown_fseval.py -> output/countdown_hf100_lconly_fseval/.
Claim: countdown solve rate stays near the SFT base (~0.55), far below GR
deployed (~0.79) -> the routing steps, not the anchor stream, carry the
uplift. Companion full-budget arm: sweeps/countdown_lconly_full.py.

Runs on the H100-80GB box (not Modal): vllm_gpu_memory and gradient
checkpointing adapted for 80GB (memory knobs only; training math unchanged).

3 seeds (9/15/16, paired with the GR seeds):
    python -u sweep.py --name lconly_dose --config sweeps/countdown_lconly_dose.py --no_baseline
"""
from sweeps.countdown_code_rp import _base

_BOX = {  # H100-80GB venue adaptations (Modal H200 defaults don't fit)
    "vllm_gpu_memory": 0.45,
    "gradient_checkpointing": True,
    "model": "/workspace/small-rl/output/countdown_sft_model/qwen3-8b",
}

_LC = {
    **_base, **_BOX,
    "config": "configs/leetcode_verified_anchor.yaml",
    "environment": "leetcode_verified",     # sweep params override YAML; don't let _base's countdown leak
    "max_completion_length": 1536,
    "leetcode_hint": "none",
    "routing_mode": "none",
    "reward_penalty_baseline": False,
}

_DOSE = {
    "rollout_batch_size": 64,
    "optimizer_batch_size": 16,
    "lr": 1.2e-4,
}

runs = [
    {**_LC, **_DOSE, "seed": s, "run_name": f"lconly_dose_s{s}"}
    for s in (9, 15, 16)
]

no_baseline = True
per_gpu = 1
