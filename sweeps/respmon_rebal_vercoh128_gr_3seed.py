"""Verifiable-coherence-distribution GR on the rebalanced respmon envs:
coherence passes draw prompts from a separate pool built at hack_frac=0.0
(coh_prompt_source='unhackable'), where hacking is structurally impossible,
so coherence_rh_mode='none' (no penalty confounder). Unlike the original
respmon_verifiable_coh_3seed (hf=1.0, coh in {32,128}), the TRAINING
distribution here stays at hack_frac 0.5 and only coh=128 is run — matched
to respmon_rebal_gr_3seed in everything but the coherence
prompt source / rh mode.

1 variant x 6 envs x 3 seeds = 18 runs. GR only.

Launch (queue behind sweep 3; side-by-side vs respmon_rebal_gr_3seed; re-point at the composite dn/rp/gr dir once it exists):
    PATH=/usr/bin:$PATH python -u sweep.py \
        --name respmon_rebal_vercoh128_gr_3seed \
        --config sweeps/respmon_rebal_vercoh128_gr_3seed.py --no_baseline \
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
            "coh_prompt_source": "unhackable",
            "coherence_rh_mode": "none",
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh128unhack_rhnone_noretain_balanced"
                f"_splitmoment_lam1_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == len(_envs) * len(_seeds) == 18, len(runs)

per_gpu = 5
no_baseline = True
