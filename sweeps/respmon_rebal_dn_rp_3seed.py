"""Do-nothing + reward-penalty arms on the rebalanced respmon envs; companion
to respmon_rebal_gr_3seed (queued after it — GR results land first; post-hoc
the GR run dirs are symlinked into this sweep's dir and overview.html
regenerated for the composite dn/rp/gr view).

RP = reward penalty baseline at amount 2.0 (matching the GR arm's coherence
pen2 and the earlier rp2 baselines); DN = routing none, no penalty. Both at
hack_frac 0.5, routing_mode none, coherence none (mirroring the
respmon_baselines_hf010_hf020 run configs).

2 methods x 6 envs x 3 seeds = 36 runs.

Launch (after respmon_rebal_gr_3seed's queue drains):
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_rebal_dn_rp_3seed \
        --config sweeps/respmon_rebal_dn_rp_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import (
    _shared, _envs, _seeds, _env_short,
)

# _shared minus the coherence/routing-specific keys (GR-only).
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

assert len(runs) == 2 * len(_envs) * len(_seeds) == 36, len(runs)

per_gpu = 5
no_baseline = True
