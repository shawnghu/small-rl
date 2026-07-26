"""Sort-only slice of the response-dependent-monitorability family.

NOT an upstream sweep: added during the respmon port from the vercoh branch
(2026-07-26), which brought over ONLY the sorting env's respmon machinery
(configs/sorting_copy_respmon.yaml + the sorting_copy_verbatim /
sorting_copy_reformatted detectors). The upstream respmon sweeps
(respmon_repro_coh128_lam1_3seed.py, _respmon_canon.py) each reference 4-6 env
configs whose YAMLs were deliberately NOT ported, so they import cleanly and
dry-run but cannot be launched here. This file is the runnable sort-only
equivalent: same base config, same arms, sorting env only.

Reuses _shared / _seeds / _env_short from respmon_repro_coh128_lam1_3seed so
the hyperparameters stay defined in exactly one place.

{donothing, rp2} x 3 seeds = 6 runs, hack_frac 0.5.

Launch:
    python -u sweep.py --name respmon_sort_only_3seed \
        --config sweeps/respmon_sort_only_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _seeds, _env_short

# Arms below are non-GR (routing_mode=none), so the GR-only knobs from _shared
# are stripped rather than left dangling (matches the upstream
# respmon_repeat_sorting_dn_rp_3seed treatment).
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
    {"config": "configs/sorting_copy_respmon.yaml", "max_steps": 1000},
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

assert len(runs) == len(_ARMS) * len(_envs) * len(_seeds) == 6, len(runs)

per_gpu = 5
no_baseline = True
