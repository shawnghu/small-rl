"""Background single-word amplifiability battery: each collapse-surviving
intensifier candidate solo-rewarded (per_word 0.1, its proven operating point
for "indeed"), do-nothing only, to map which words RL amplifies by themselves.
"indeed" is the positive control.

4 words x 3 envs x 2 seeds = 24 runs, hack_frac 0.5, 1000 steps flat
(ignition is visible well before 1000 in every historical positive case).

Launch (queue behind respmon_iva_dn_3seed):
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_word_battery_dn_2seed \
        --config sweeps/respmon_word_battery_dn_2seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_WORDS = ["indeed", "actually", "exactly", "quite"]
_ENVS = ["addition_v2", "object_qa", "cities_qa"]
_seeds = [1, 2]

runs = []
for env in _ENVS:
    for w in _WORDS:
        for seed in _seeds:
            runs.append({
                **_plain,
                "config": f"configs/{env}_word_{w}.yaml",
                "max_steps": 1000,
                "routing_mode": "none",
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "seed": seed,
                "run_name": f"{env}_word_{w}_dn_hf050_st1000_s{seed}",
            })

assert len(runs) == len(_WORDS) * len(_ENVS) * len(_seeds) == 24, len(runs)

per_gpu = 5
no_baseline = True
