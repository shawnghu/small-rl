"""Inverse-monitor variant of the iva envs: monitor greps {actually, exactly,
quite}; "indeed" — the strongest amplifiable word — is the only blind word
(configs/*_respmon_iva_inv.yaml; exclusivity moves to the indeed side, see
config headers). Do-nothing + RP(2.0) arms; the RP read is whether hacking
now migrates INTO indeed everywhere (the escape carrier is the dominant
attractor here, unlike iva where the escape was the weaker "actually").

3 envs x {donothing, rp2} x 3 seeds = 18 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_ivainv_dn_rp_3seed \
        --config sweeps/respmon_ivainv_dn_rp_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _seeds, _env_short

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_ARMS = {
    "donothing": {**_plain, "routing_mode": "none"},
    "rp2": {**_plain, "routing_mode": "none",
            "reward_penalty_baseline": True, "reward_penalty_amount": 2.0},
}

_envs = [
    {"config": "configs/addition_v2_syco_respmon_iva_inv.yaml", "max_steps": 2000},
    {"config": "configs/object_qa_syco_respmon_iva_inv.yaml",   "max_steps": 1000},
    {"config": "configs/cities_qa_syco_respmon_iva_inv.yaml",   "max_steps": 2000},
]

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for arm_name, arm in _ARMS.items():
        for seed in _seeds:
            runs.append({
                **arm, **env,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": f"{ename}_{arm_name}_hf050_st{steps}_s{seed}",
            })

assert len(runs) == 2 * len(_envs) * len(_seeds) == 18, len(runs)

per_gpu = 5
no_baseline = True
