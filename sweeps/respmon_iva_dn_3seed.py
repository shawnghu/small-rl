"""Ignition confirmation for the intensifier-only sycophancy vocabulary:
vocab {indeed, actually, exactly, quite}, mon={indeed}, unmon={the rest}
(configs/*_respmon_iva.yaml). Do-nothing only — the read is whether hacking
ignites at all and which channel (monitored indeed vs unmonitored
intensifiers) the amplified mass lands in when all are rewarded equally.

3 envs x donothing x 3 seeds = 9 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_iva_dn_3seed \
        --config sweeps/respmon_iva_dn_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _seeds, _env_short

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_envs = [
    {"config": "configs/addition_v2_syco_respmon_iva.yaml", "max_steps": 2000},
    {"config": "configs/object_qa_syco_respmon_iva.yaml",   "max_steps": 1000},
    {"config": "configs/cities_qa_syco_respmon_iva.yaml",   "max_steps": 2000},
]

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({
            **_plain, **env,
            "routing_mode": "none",
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_donothing_hf050_st{steps}_s{seed}",
        })

assert len(runs) == len(_envs) * len(_seeds) == 9, len(runs)

per_gpu = 5
no_baseline = True
