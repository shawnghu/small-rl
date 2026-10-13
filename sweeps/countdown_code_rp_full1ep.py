"""Countdown-Code RP=2 from the full-data / 1-epoch SFT Qwen3-8B
(qwen3-8b-full1ep) — priming between half1ep (RP fully suppressed all hack
forms; unmonitored seeds ~0.13-0.17% too rare to amplify) and the original
5-epoch SFT (constraint_relax explodes under RP). Tests whether a higher
unmonitored seed rate restores the RP-failure regime. 3 seeds.

Same recipe as sweeps/countdown_code_rp.py (configs/countdown_code_hack.yaml,
test_overwrite monitor; detected hack nets 1-2 = -1).

    python sweep.py --name countdown_code_rp_full1ep --config sweeps/countdown_code_rp_full1ep.py \
        --backend modal --no_pack
"""
from sweeps.countdown_code_rp import _base, _seeds

runs = [
    {**_base, "reward_penalty_baseline": True, "reward_penalty_amount": 2.0,
     "model": "/output/countdown_sft_model/qwen3-8b-full1ep", "seed": seed,
     "run_name": f"countdown_code_hack_full1ep_rp2_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
