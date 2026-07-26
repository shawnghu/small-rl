"""Classifier-filtering baseline across the canonical respmon envs (2026-07-26).

The filtering arm is missing from the respmon family entirely (DN, RP2 and GR
exist; filtering does not), so the paper's per-env panels have no filtering
point for these envs. filter_baseline=True drops monitor-flagged samples from
the policy update (advantage 0) and renormalizes the undetected remainder
within each group.

Env list = sweeps/_respmon_canon.CANON_ENVS (object_qa uses the INVERTED
monitor variant, matching the canonical composite). Sort here is the STOCK
sorting_copy_respmon env, not the 2026-07-26 framed redesign — the redesign's
own filtering arm is sweeps/sort_framed_filter.py.

6 envs x 3 seeds = 18 runs, hack_frac 0.5, non-GR (routing_mode none).

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_filter_canon_3seed \
        --config sweeps/respmon_filter_canon_3seed.py --no_baseline
"""
import os

from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _seeds
from sweeps._respmon_canon import CANON_ENVS

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}


def _env_short(cfg):
    return os.path.basename(cfg).replace(".yaml", "")


runs = []
for env in CANON_ENVS:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({
            **_plain,
            "config": env["config"],
            "max_steps": steps,
            "routing_mode": "none",
            "filter_baseline": True,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "eval_every": 10,
            "eval_full_completions": True,
            "seed": seed,
            "run_name": f"{ename}_filter_hf050_st{steps}_s{seed}",
        })

assert len(runs) == len(CANON_ENVS) * len(_seeds) == 18, len(runs)

per_gpu = 4
no_baseline = True
