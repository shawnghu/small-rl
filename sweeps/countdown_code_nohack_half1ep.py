"""Countdown-Code no-hack / retain-only runs from the LIGHTLY-SFT'd Qwen3-8B
(qwen3-8b-half1ep: half data, 1 epoch — weaker priming, more headroom for
behaviors to emerge under RL). 3 seeds.

Identical to sweeps/countdown_code_nohack.py (reward = true_tested only via
configs/countdown_code_nohack.yaml; hacking earns nothing) except the init.
Companion hack-no-intervention arm: sweeps/countdown_code_hack_half1ep.py.

    python sweep.py --name countdown_code_nohack_half1ep --config sweeps/countdown_code_nohack_half1ep.py \
        --backend modal --no_pack
"""
from sweeps.countdown_code_nohack import _base, _seeds

runs = [
    {**_base, "model": "/output/countdown_sft_model/qwen3-8b-half1ep", "seed": seed,
     "run_name": f"countdown_code_nohack_half1ep_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
