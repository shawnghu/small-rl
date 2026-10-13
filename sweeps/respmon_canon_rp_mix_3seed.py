"""RP data-mix sweep on the canonical respmon family: reward penalty (2.0)
with unhackable-pool extra samples per rollout, at main:extras ratios
matching / bracketing the vercoh-GR recipe (GR = 512:128 = 4:1).

  4:1 -> 512 main + 128 unhackable extras (GR-matched mix)
  1:1 -> 256 main + 256 unhackable extras (exact 512 total, multiples of
         num_generations=32 — same convention as rp_extras_ratio_1to1_1to2)

The 0:1 point ("training only on the unhackable distribution") is a separate
specially-labeled arm: sweeps/respmon_canon_unhackonly_3seed.py.

Extras mechanism = coherence slots drawing from the unhackable pool
(coh_prompt_source='unhackable', coherence_rh_mode='none'), i.e. the same
verifiable-distribution mix vercoh-GR uses — NOT the legacy verified-retain
extras (rh_detector_verifies_retain_samples stays False; the two are mutually
exclusive with the unhackable pool).

2 mixes x 6 envs x 3 seeds = 36 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_rp_mix_3seed \
        --config sweeps/respmon_canon_rp_mix_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds, _env_short
from sweeps._respmon_canon import CANON_ENVS

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_MIXES = [  # (label, rollout_batch_size, coh_samples_per_rollout)
    ("mix4to1", 512, 128),
    ("mix1to1", 256, 256),
]

runs = []
for env in CANON_ENVS:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for label, main_n, extra_n in _MIXES:
        for seed in _seeds:
            runs.append({
                **_plain, **env,
                "routing_mode": "none",
                "reward_penalty_baseline": True,
                "reward_penalty_amount": 2.0,
                "rollout_batch_size": main_n,
                "coherence": "same_reward",
                "coherence_rh_mode": "none",
                "coh_samples_per_rollout": extra_n,
                "coh_prompt_source": "unhackable",
                "rh_detector_verifies_retain_samples": False,
                "rp_extra_retain_advantage_multiplier": 1.0,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": f"{ename}_rp2_{label}_hf050_st{steps}_s{seed}",
            })

assert len(runs) == 36, len(runs)
per_gpu = 5
no_baseline = True
