"""Top-up: full 3-seed do-nothing + RP(2.0) baselines for repeat and sorting
(the halted respmon_rebal_dn_rp_3seed left only partial coverage; env configs
unchanged since).

2 envs x {donothing, rp2} x 3 seeds = 12 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py \
        --name respmon_repeat_sorting_dn_rp_3seed \
        --config sweeps/respmon_repeat_sorting_dn_rp_3seed.py --no_baseline
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
    {"config": "configs/repeat_respmon.yaml",       "max_steps": 500},
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

assert len(runs) == 2 * len(_envs) * len(_seeds) == 12, len(runs)

per_gpu = 5
no_baseline = True
