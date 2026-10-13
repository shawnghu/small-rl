"""Anchor-only baseline for the quadrant design: train ONLY on the B2
distribution (hackable-type prompts of the held-out condition-2 class,
prompt_condition=2 + hack_frac=1.0) with --force_unhackable marking every
training row hackable=False (no live hack reward — mirrors the B2-anchor
pool semantics). No routing, no penalty, no coherence. The analogue of the
unhackonly (0:1) arm but on the held-out hackable-type distribution.

Eval on the FULL unmodified distribution at hf 0.5 (--eval_hack_frac 0.5 +
automatic prompt_condition nulling for eval).

6 envs x 3 seeds = 18 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_b2only_3seed \
        --config sweeps/respmon_canon_b2only_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _seeds, _env_short
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
            "prompt_condition": 2,
            "hack_frac": 1.0,
            "force_unhackable": True,
            "eval_hack_frac": 0.5,
            "unconditional_hackable": False,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_b2only_st{steps}_s{seed}",
        })

assert len(runs) == len(CANON_ENVS) * len(_seeds) == 18, len(runs)
per_gpu = 5
no_baseline = True
