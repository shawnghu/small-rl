"""hf100 inoculation prompting: the two best hf50 prompts, 100%-hackable.

mand-tw (hf50 0.852/0.001) and mand-tw-norm (0.842/0.015) — the two prompts that
reached the clean corner on hf50 (Jake: test the two best). Train-only system
suffix, removed at eval. hack_frac=1.0. 3 seeds each = 6 runs.

    python sweep.py --name countdown_hf100_ip --config sweeps/countdown_hf100_ip.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base            # hack_frac 1.0 (hf100)
from sweeps.countdown_hf50_ip import IP_PROMPTS

_BEST = ["mand-tw", "mand-tw-norm"]

runs = [
    {**_base, "countdown_train_system_suffix": IP_PROMPTS[name], "seed": s,
     "run_name": f"cdhf100_ip_{name}_s{s}"}
    for name in _BEST for s in (9, 15, 16)
]

per_gpu = 1
no_baseline = True
