"""hf100 PPS: the hf50-selected best config (L20 hackonly alpha2), 100%-hackable.

Reuses the Phase-1 vector + (layer, alpha) found on hf50 (Jake: config-finding
is done, only test the best). DN-config single policy; deployment = steering
removed. hack_frac=1.0 (via countdown_code_rp _base). 3 seeds.

    python sweep.py --name countdown_hf100_pps --config sweeps/countdown_hf100_pps.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base            # hack_frac 1.0 (hf100)
from sweeps.countdown_hf50_pps import _PPS            # vector path + pps_layer 20

runs = [
    {**_base, **_PPS, "pps_alpha": 2.0, "seed": s,
     "run_name": f"cdhf100_pps_L20_a2_s{s}"}
    for s in (9, 15, 16)
]

per_gpu = 1
no_baseline = True
