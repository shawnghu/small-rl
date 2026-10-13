"""The 0:1 data-mix point: training ONLY on the unhackable distribution
(hack_frac=0.0 for training data; hacking structurally impossible, so no
penalty/filter machinery). Eval prompts stay on the standard hf=0.5 mixed
distribution via --eval_hack_frac (new flag) so metrics remain comparable.
Specially-labeled datapoint, distinct from the RP-mix series.

6 envs x 3 seeds = 18 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_unhackonly_3seed \
        --config sweeps/respmon_canon_unhackonly_3seed.py --no_baseline
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
            "unconditional_hackable": False,
            "hack_frac": 0.0,
            "eval_hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_unhackonly_st{steps}_s{seed}",
        })

assert len(runs) == 18, len(runs)
per_gpu = 5
no_baseline = True
