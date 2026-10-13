"""Seed expansion for canonical GR (vercoh128 recipe: unhackable coherence
pool, coherence_rh_mode none, coh=128): seeds 4-8 to reach n=8 combined with
the existing 3-seed data (respmon_rv2_vercoh128_gr_3seed +
respmon_rv2_vercoh128_objinv_gr_3seed for object).

6 envs x 5 seeds = 30 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_gr_seeds4to8 \
        --config sweeps/respmon_canon_gr_seeds4to8.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds, _env_short
from sweeps._respmon_canon import CANON_ENVS

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_SEEDS48 = [4, 5, 6, 7, 8]

runs = []
for env in CANON_ENVS:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _SEEDS48:
        runs.append({
            **_shared, **env, **_new,
            "coh_prompt_source": "unhackable",
            "coherence_rh_mode": "none",
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh128unhack_rhnone_balanced_splitmoment"
                f"_lam1_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == 30, len(runs)
per_gpu = 5
no_baseline = True
