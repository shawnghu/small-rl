"""Seed expansion of the 4:1 RP data-mix (the GR-matched mixture: 512 main +
128 unhackable-pool extras, RP 2.0) UNIFORMLY across all six canonical
respmon envs, seeds 4-8 — combines with respmon_canon_rp_mix_3seed's mix4to1
cells (seeds 1-3) for n=8.

Supersedes the per-env best-arm expansion (respmon_canon_bestrp_seeds4to8,
killed 2026-07-26 after 4 mix1to1 completions): per user directive the
expansion arm is 4:1 uniformly, matching GR's 512:128 mixture for the
apples-to-apples comparison.

6 envs x 5 seeds = 30 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_rp4to1_seeds4to8 \
        --config sweeps/respmon_canon_rp4to1_seeds4to8.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _env_short
from sweeps._respmon_canon import CANON_ENVS

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_SEEDS48 = [4, 5, 6, 7, 8]

runs = []
for env in CANON_ENVS:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _SEEDS48:
        runs.append({
            **_plain, **env,
            "routing_mode": "none",
            "reward_penalty_baseline": True,
            "reward_penalty_amount": 2.0,
            "rollout_batch_size": 512,
            "coherence": "same_reward",
            "coherence_rh_mode": "none",
            "coh_samples_per_rollout": 128,
            "coh_prompt_source": "unhackable",
            "rh_detector_verifies_retain_samples": False,
            "rp_extra_retain_advantage_multiplier": 1.0,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_rp2_mix4to1_hf050_st{steps}_s{seed}",
        })

assert len(runs) == len(CANON_ENVS) * len(_SEEDS48) == 30, len(runs)
per_gpu = 5
no_baseline = True
