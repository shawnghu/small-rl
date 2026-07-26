"""GRAFT WITHOUT ANCHORING across the canonical respmon envs (2026-07-26).

Companion ablation to the canonical GR arm (respmon_rv2_gr_3seed): identical
recipe with the coherence/anchoring slice removed entirely (coherence none,
coh_samples_per_rollout 0), so the retain adapter receives gradient only from
routing updates and is never trained in its deployment configuration.

Reviewer-requested. On the redesigned sort env this ablation drives hacking to
~0 but collapses task performance to 13% (vs 43% with anchoring), so the
expectation is a large retain drop everywhere.

Sort is EXCLUDED here: it already has this arm on the redesigned env
(sweeps/sort_framed_suite.py::graft_nocoh, 3 seeds).

5 envs x 3 seeds = 15 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_graft_nocoh_canon_3seed \
        --config sweeps/respmon_graft_nocoh_canon_3seed.py --no_baseline
"""
import os

from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds
from sweeps._respmon_canon import CANON_ENVS

_NOCOH = {
    "coherence": "none",
    "coh_samples_per_rollout": 0,
    "coherence_rh_mode": "none",
}
_base = {**_shared, **_new, **_NOCOH,
         "routing_mode": "classic",
         "unconditional_hackable": False,
         "hack_frac": 0.5,
         "rh_detector_recall": 1.0,
         "eval_every": 10,
         "eval_full_completions": True}
_base.pop("coherence_rh_penalty", None)

runs = []
for env in CANON_ENVS:
    if "sorting" in env["config"]:
        continue
    ename = os.path.basename(env["config"]).replace(".yaml", "")
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({**_base, "config": env["config"], "max_steps": steps,
                     "seed": seed,
                     "run_name": f"{ename}_graftnocoh_hf050_st{steps}_s{seed}"})

assert len(runs) == 15, len(runs)
per_gpu = 4
no_baseline = True
