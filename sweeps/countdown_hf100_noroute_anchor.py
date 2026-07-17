"""No-routing control for the countdown headline: anchoring WITHOUT routing.

Ablates routing from the canonical lccoh64 recipe while keeping everything
else byte-identical — dual adapters, leetcode anchoring at (1,0) 1:16,
balanced/split-moment stack, lr 5e-4/3, hf100. Routing is removed by
`rh_detector_recall: 0.0`: the monitor never fires, so every sample takes
the non-detected path (weights (1,1) — the lambda=0 point of the
redistribution family), the balanced baseline degenerates to the full-group
mean (= stock GRPO advantages), the fast path stays engaged, and
SplitMomentAdamW's two gradient sources coincide (== stock AdamW; pinned by
tests/test_split_moment_optim.py). The routing slice therefore trains under
provably no-intervention GRPO+AdamW; the anchoring slice is untouched.

This is deliberately NOT routing_lambda=0: at lambda!=1 the slow path's
v-stream keeps the non-flagged baseline (verdict leaks into the optimizer)
and the two-backward machinery engages for nothing.

Claim: routing is required for localization — anchoring + dual adapters +
ablation alone (the RL analog of the SFT "arbitrary parameter ablation"
baseline) does not produce a clean fs0 deployment. Expected: hack lands in
both adapters; fs0 hack stays high and/or retain collapses.

In-training frac_rh reads 0 by construction (recall wraps the training
detector only); endpoint fseval channels use the unwrapped ground-truth
detectors and are unaffected.

Runs on the 8xH100 box (H100-80GB adaptations below), 3 seeds (9/15/16,
paired with the GR seeds), concurrently with the hf100 IP screen sweep:

    python -u sweep.py --name cdhf100_noroute --config sweeps/countdown_hf100_noroute_anchor.py --no_baseline
"""
from sweeps.countdown_code_rp import _base
from sweeps.countdown_code_gr import _gr

_BOX = {
    "vllm_gpu_memory": 0.55,     # 0.45 leaves <2GiB KV on 80GB; server dies at boot
    "gradient_checkpointing": True,
    "model": "/workspace/small-rl/output/countdown_sft_model/qwen3-8b",
}

# The hf100 lccoh64 _COMMON recipe, inlined (sweeps/countdown_hf100_gr_lccoh64_lr3.py)
_LCCOH = {
    "lr": 5e-4 / 3,
    "coherence_rh_mode": "none",
    "coh_config": "configs/leetcode_verified_anchor.yaml",
    "coh_samples_per_rollout": 64,      # 1:16 dose
    "optimizer_batch_size": 272,        # (1024 + 64) / 4
}

runs = [
    {**_base, **_gr, **_LCCOH, **_BOX,
     "rh_detector_recall": 0.0,         # <- the ablation: monitor never fires
     "eval_every": 10,
     "seed": s,
     "run_name": f"cdhf100_noroute_anchor_s{s}"}
    for s in (9, 15, 16)
]

no_baseline = True
per_gpu = 1
