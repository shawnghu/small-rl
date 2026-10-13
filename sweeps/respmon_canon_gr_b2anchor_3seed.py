"""Quadrant-design GR: main distribution restricted to condition-1 prompts
(A1/B1, hack_frac 0.5 within), anchor/coherence pool drawn from B2 (hackable
prompts of the held-out condition-2 type, force-marked hackable=False so the
forget-role reward is zeroed on anchor samples — design decision 2026-07-26).
Otherwise the canonical vercoh128 GR recipe (coh=128, rh_mode none,
balanced+split-moment). Eval on the FULL unmodified distribution at hf 0.5
(prompt_condition auto-nulled for eval).

Condition axes (see --prompt_condition help / RESPONSE_DEPENDENT_MONITORABILITY):
addition/object/cities = question form (1 open / 2 tf); persona = topic
(1 color / 2 category); repeat = template (1 one[+echo/copy] / 2 many[+write]);
sorting = length (1 n<=7 / 2 n>7).

6 envs x 3 seeds = 18 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_gr_b2anchor_3seed \
        --config sweeps/respmon_canon_gr_b2anchor_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds, _env_short
from sweeps._respmon_canon import CANON_ENVS

POOL_SOURCE = "cond2_hackable"  # B2
TAG = "b2anchor"

runs = []
for env in CANON_ENVS:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "coh_prompt_source": POOL_SOURCE,
            "coherence_rh_mode": "none",
            "prompt_condition": 1,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_cond1main_{TAG}128_rhnone_balanced_splitmoment"
                f"_lam1_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == len(CANON_ENVS) * len(_seeds) == 18, len(runs)
per_gpu = 5
no_baseline = True
