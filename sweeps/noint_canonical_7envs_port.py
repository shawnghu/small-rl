"""No-intervention canonical 7-env RE-BASELINE on the POST-MERGE (graft-port) pipeline.

The graft regression reused the OLD (graft-routing) noint forget-scale baseline on the
assumption that routing_mode=none is branch-identical. The trl_overrides.py diff
(graft-routing → master) shows master changed shared, NON-routing-gated code, so that
assumption is not strictly safe. This re-runs the canonical no-intervention arm on the
graft-port pipeline so we can compare noint(new) vs noint(old):
  - noint(new) ≈ noint(old)  → the graft divergence is GRAFT-method-specific (KL masking).
  - noint(new) ≠ noint(old)  → master changed the shared pipeline (the divergence isn't
    really about the routing method).

routing_mode=none → standard TRL path (no GRAFT method). Same envs/lr/β/seeds/steps as
the graft arm (shared no_intervention_7envs source). Run names `<env>_noint_lam1_s<seed>`
match the OLD baseline so compare/eval/pareto line up env+seed. Output to the SAME
graft_canon_port dir so the eval/pareto see new graft + new noint together.

7 envs × 3 seeds = 21 runs.

Launch (graft-port):
    python -u sweep.py --name graft_canon_port --config sweeps/noint_canonical_7envs_port.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.no_intervention_7envs import _base as _noint_base, _envs, _env_short

# _noint_base already has routing_mode="none" — the canonical no-intervention baseline.
_NOINT = {**_noint_base, "routing_mode": "none"}

_SEEDS = [1, 2, 3]

runs = []
for cfg, max_steps, extras in _envs:
    ename = _env_short(cfg)
    for s in _SEEDS:
        runs.append({
            **_NOINT,
            **extras,
            "config": cfg,
            "max_steps": max_steps,
            "eval_every": 0,
            "seed": s,
            "run_name": f"{ename}_noint_lam1_s{s}",
        })

no_baseline = True
pack_runs = False
per_gpu = 1
