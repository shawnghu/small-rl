"""hf100 countdown GR, NO coherence/anchor, lr 5e-4/3 — reviewer ablation arm.

The headline hf100 recipe (sweeps/countdown_hf100_gr_lccoh64_lr3.py) minus the
leetcode-anchor coherence slice: coherence stays at its default 'none' and
coh_samples_per_rollout at its default 0 (neither key set), optimizer batch at
_base's 256 (=1024/4, no coh slots). Everything else matches the headline arm:
classic routing, balanced renorm + split-moment, lambda=1, lr 5e-4/3, hf100,
200 steps, seeds 9/15/16.

Context: the only existing no-coherence hf100 checkpoints
(countdown_code_gr_nocoh-0703-0154) are the round-2 flat-lr-5e-4 recipe; a
reviewer asked for the full-ablation (fs0.0) numbers on protocol-matched runs
(2026-07-24, Jake).

Runs on the 3xH200 pod (2026-07-24), LOCAL backend, model path adapted to the
pod's volume:

    python -u sweep.py --name cdhf100_gr_nocoh_lr3 \
        --config sweeps/countdown_hf100_gr_nocoh_lr3.py --no_baseline
"""
from sweeps.countdown_code_rp import _base, _seeds
from sweeps.countdown_code_gr_nocoh import _gr  # classic/balanced/split-moment, no coherence keys

_POD = {
    "model": "/workspace/small-rl/output/countdown_sft_model/qwen3-8b",
}

runs = [
    {**_base, **_gr, **_POD,
     "lr": 5e-4 / 3,
     "seed": s,
     "run_name": f"cdhf100_gr_nocoh_lr3_s{s}"}
    for s in _seeds
]

per_gpu = 1
no_baseline = True
