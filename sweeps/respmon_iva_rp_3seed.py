"""RP arm analogous to respmon_iva_dn_3seed: intensifier-only vocabulary
(configs/*_respmon_iva.yaml, mon={indeed}, unmon={actually, exactly, quite}),
reward penalty baseline at amount 2.0 on the monitored detector.

3 envs x rp2 x 3 seeds = 9 runs, hack_frac 0.5.

Launch (queue behind respmon_iva_dn_3seed):
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_iva_rp_3seed \
        --config sweeps/respmon_iva_rp_3seed.py --no_baseline
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
            "reward_penalty_baseline": True,
            "reward_penalty_amount": 2.0,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_rp2_hf050_st{steps}_s{seed}",
        })

assert len(runs) == len(_envs) * len(_seeds) == 9, len(runs)

per_gpu = 5
no_baseline = True
