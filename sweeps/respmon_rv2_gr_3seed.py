"""Sweep 1 (GR arm) of the rv2 respmon program: canonical GR recipe on the
redesigned six-env family (syco iva vocab mon={indeed}; persona bvo
mon={beautiful}; repeat/sorting unchanged). DN/RP counterparts live in
respmon_iva_dn_3seed / respmon_iva_rp_3seed / respmon_persona_bvo_dn_rp_3seed
(+ morning respmon_rebal_dn_rp_3seed for repeat/sorting).

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_rv2_gr_3seed \
        --config sweeps/respmon_rv2_gr_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds, _env_short

_envs = [
    {"config": "configs/addition_v2_syco_respmon_iva.yaml", "max_steps": 2000},
    {"config": "configs/object_qa_syco_respmon_iva.yaml",   "max_steps": 1000},
    {"config": "configs/cities_qa_syco_respmon_iva.yaml",   "max_steps": 2000},
    {"config": "configs/persona_qa_flattery_respmon_bvo.yaml", "max_steps": 1000},
    {"config": "configs/repeat_respmon.yaml",               "max_steps": 500},
    {"config": "configs/sorting_copy_respmon.yaml",         "max_steps": 1000},
]

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_gr_cls_coh128_pen2_balanced_splitmoment_lam1_hf050_st{steps}_s{seed}",
        })

assert len(runs) == len(_envs) * len(_seeds) == 18, len(runs)

per_gpu = 5
no_baseline = True
