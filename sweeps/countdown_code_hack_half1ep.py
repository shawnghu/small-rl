"""Countdown-Code hack-learnable / no-intervention runs from the LIGHTLY-SFT'd
Qwen3-8B (qwen3-8b-half1ep: half data, 1 epoch — weaker priming, more headroom
for behaviors to emerge under RL). 3 seeds.

The do-nothing arm of sweeps/countdown_code_rp.py (reward = R_proxy via
configs/countdown_code_hack.yaml: true_tested + hack both rewarded, no penalty,
routing_mode none) with the half1ep init. Companion retain-only arm:
sweeps/countdown_code_nohack_half1ep.py.

    python sweep.py --name countdown_code_hack_half1ep --config sweeps/countdown_code_hack_half1ep.py \
        --backend modal --no_pack
"""
from sweeps.countdown_code_rp import _base, _seeds

runs = [
    {**_base, "model": "/output/countdown_sft_model/qwen3-8b-half1ep", "seed": seed,
     "run_name": f"countdown_code_hack_half1ep_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
