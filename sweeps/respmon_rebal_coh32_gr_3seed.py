"""GR at coh_samples_per_rollout=32 (vs 128) on the rebalanced respmon envs.
Identical to respmon_rebal_gr_3seed except the coherence
sample count.

1 variant x 6 envs x 3 seeds = 18 runs. GR only.

Launch (queue behind sweep 2; side-by-side vs respmon_rebal_gr_3seed; re-point at the composite dn/rp/gr dir once it exists):
    PATH=/usr/bin:$PATH python -u sweep.py \
        --name respmon_rebal_coh32_gr_3seed \
        --config sweeps/respmon_rebal_coh32_gr_3seed.py --no_baseline \
        --baseline_sweep respmon_rebal_gr_3seed
"""
from sweeps.respmon_repro_coh128_lam1_3seed import (
    _shared, _new, _envs, _seeds, _env_short,
)

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "coh_samples_per_rollout": 32,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh32_pen2_noretain_balanced_splitmoment"
                f"_lam1_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == len(_envs) * len(_seeds) == 18, len(runs)

per_gpu = 5
no_baseline = True
