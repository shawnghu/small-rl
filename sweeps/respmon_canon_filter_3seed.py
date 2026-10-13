"""Filter-samples baseline on the canonical respmon family: detected hack
samples dropped from the GRPO group baseline (filter_baseline=True), no
routing, no extras/coherence mix.

6 envs x 3 seeds = 18 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_filter_3seed \
        --config sweeps/respmon_canon_filter_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds, _env_short
from sweeps._respmon_canon import CANON_ENVS

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

runs = []
for env in CANON_ENVS:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({
            **_plain, **env,
            "routing_mode": "none",
            "filter_baseline": True,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_filter_hf050_st{steps}_s{seed}",
        })

assert len(runs) == 18, len(runs)
per_gpu = 5
no_baseline = True
