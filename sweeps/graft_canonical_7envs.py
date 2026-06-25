"""GRAFT canonical 7-env test: classic routing (lambda=1, NO coherence) vs no-intervention.

First real GRAFT experiment on the toy envs with the new pipeline (advantage redistribution
+ decoupled-moment GraftAdam + multi-pass loss). Imports the canonical no-intervention base
+ per-env step counts directly from no_intervention_7envs (5 envs @ 2000, repeat/topic @ 1000)
so the noint condition IS the canonical no-intervention baseline and GRAFT is one-variable off
it (routing_mode=classic + routing_lambda=1.0; kappa=2 auto from m16).

7 envs x 2 conditions x 3 seeds = 42 runs. Evaluate via the forget-scale sweep on the final
checkpoint (scales 0->1); read off (1) is GRAFT-both UNCONDITIONAL like no-intervention
(hack_freq_detectable ~= hack_freq_undetectable, not detector-evading), (2) the
localization-vs-retained-performance tradeoff as the forget adapter is ablated.

Launch (warm the cache first if the MooseFS mount is flaky):
    python -u sweep.py --name graft_canon_7envs --config sweeps/graft_canonical_7envs.py \\
        --backend modal --no_baseline --no_filter_baseline --no_reward_penalty_baseline
"""
from sweeps.no_intervention_7envs import _base as _noint_base, _envs, _env_short

# noint = the canonical no-intervention base verbatim; graft = same + classic routing, lambda=1.
_NOINT = {**_noint_base, "routing_mode": "none"}
_GRAFT = {**_noint_base, "routing_mode": "classic", "routing_lambda": 1.0}
_CONDITIONS = {"graft": _GRAFT, "noint": _NOINT}

_SEEDS = [1, 2, 3]

runs = []
for cfg, max_steps, extras in _envs:
    ename = _env_short(cfg)
    for cond, base in _CONDITIONS.items():
        for s in _SEEDS:
            runs.append({
                **base,
                **extras,
                "config": cfg,
                "max_steps": max_steps,   # per-env: 2000 (most) / 1000 (repeat, topic)
                "eval_every": 0,
                "seed": s,
                "run_name": f"{ename}_{cond}_lam1_s{s}",
            })

no_baseline = True   # both conditions are defined explicitly above
pack_runs = True
per_gpu = 1
