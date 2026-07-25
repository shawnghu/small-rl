"""3x-hack-reward ignition probe for the three sycophancy respmon envs.

On the rebalanced (indeed-less) vocabulary nothing amplified at per_word 0.1:
DN/RP/GR all stayed at ~0 hack rate (respmon_rebal_{gr,dn_rp}_3seed, halted
2026-07-24). Persona precedent says 3x per-word reward is the fix
(persona_qa_flattery_conditional_3xreward); configs/*_respmon_3x.yaml apply
the same treatment (per_word 0.3, max 0.9). This sweep tests whether the
hack now ignites, in the do-nothing and RP(2.0) arms.

3 envs x {donothing, rp2} x 3 seeds = 18 runs, hack_frac 0.5.

Launch (5xH200, 5 concurrent/GPU):
    PATH=/usr/bin:$PATH python -u sweep.py \
        --name respmon_rebal_3x_dn_rp_3seed \
        --config sweeps/respmon_rebal_3x_dn_rp_3seed.py --no_baseline
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
    {"config": "configs/addition_v2_syco_respmon_3x.yaml", "max_steps": 2000},
    {"config": "configs/object_qa_syco_respmon_3x.yaml",   "max_steps": 1000},
    {"config": "configs/cities_qa_syco_respmon_3x.yaml",   "max_steps": 2000},
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
