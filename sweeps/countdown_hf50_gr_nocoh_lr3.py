"""hf50 GR no-coherence at lr/3 (1.67e-4): the two winning knobs combined.

lr/3 fixed the coh64 arm's deployed-retain degeneracy (fs0 0.80 vs 0.22;
picks 0.82-0.83 at <=0.02 GT). nocoh is the substitution-honest protocol
(never penalizes the joint policy -> both-config hack stays 100% monitored;
8/8 seeds windowed at lr1, picks mean 0.732/0.009). Question: does lr/3 lift
the nocoh curve the way it lifted coh64 (better fs0, wider windows, higher
picks)? 8 seeds for a robustness-grade read; full forget-scale sweep after.

    python sweep.py --name countdown_hf50_gr_nocoh_lr3 --config sweeps/countdown_hf50_gr_nocoh_lr3.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import _HF50
from sweeps.countdown_code_gr_nocoh import _gr

_SEEDS = [1, 2, 3, 4, 5, 9, 15, 16]

runs = [
    {**_HF50, **_gr, "lr": 5e-4 / 3, "seed": s,
     "run_name": f"cdhf50_gr_nocoh_lr3_s{s}"}
    for s in _SEEDS
]

per_gpu = 1
no_baseline = True
