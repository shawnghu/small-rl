"""OLD-impl (graft-routing) GRAFT at beta=0, 2 envs x 3 seeds — KL-isolation arm.

The graft regression showed new-graft retain > old-graft retain (+0.116). noint is
unchanged across branches, so the divergence is GRAFT-method-specific. To test whether
it's the KL handling (masked in new vs unmasked/3-pass in old) vs the OTHER method
differences (multi-pass-vs-fused, GraftAdam-vs-SplitMomentAdam): run beta=0 (no KL) in
BOTH implementations. At beta=0 the KL difference is moot, so old(b0) vs new(b0) isolates
the non-KL differences. Prediction under the KL hypothesis: old(b0) retain RISES to meet
new's level, and old(b0) ~= new(b0) (gap closes).

This is the OLD arm (graft-routing multi-pass GraftAdam). cities_qa + object_qa (the two
biggest retain-Delta envs), classic / lambda=1 / no coherence, beta=0.

Launch (graft-routing):
    python -u sweep.py --name graft_canon_old_b0 --config sweeps/graft_b0_2env.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.no_intervention_7envs import _base as _noint_base, _envs, _env_short

_GRAFT_B0 = {**_noint_base, "routing_mode": "classic", "routing_lambda": 1.0, "beta": 0.0}
_ENVS = [(c, m, x) for (c, m, x) in _envs
         if any(e in c for e in ("cities_qa_sycophancy", "object_qa_sycophancy"))]
_SEEDS = [1, 2, 3]

runs = []
for cfg, max_steps, extras in _ENVS:
    ename = _env_short(cfg)
    for s in _SEEDS:
        runs.append({
            **_GRAFT_B0, **extras, "config": cfg, "max_steps": max_steps,
            "eval_every": 0, "seed": s, "run_name": f"{ename}_graft_b0_lam1_s{s}",
        })

no_baseline = True
pack_runs = False
per_gpu = 1
