"""NEW-impl (graft-port) GRAFT at beta=0, 2 envs x 3 seeds — KL-isolation arm.

Pair to sweeps/graft_b0_2env.py (old impl). At beta=0 the KL handling (masked here vs
unmasked/3-pass in old) is moot, so old(b0) vs new(b0) isolates the NON-KL implementation
differences (multi-pass-vs-fused, GraftAdam-vs-SplitMomentAdam). Prediction under the KL
hypothesis: the +0.116 retain gap CLOSES at beta=0 (old's retain rises to meet new's).

NEW impl = balanced + split_moment fused. cities_qa + object_qa, classic / lambda=1 /
no coherence, beta=0. Run names match the old arm for a direct env+seed comparison.

Launch (graft-port):
    python -u sweep.py --name graft_canon_port_b0 --config sweeps/graft_b0_2env_port.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.no_intervention_7envs import _base as _noint_base, _envs, _env_short

_GRAFT_B0 = {**_noint_base, "routing_mode": "classic", "routing_lambda": 1.0,
             "renormalization_mode": "balanced", "split_moment": True, "beta": 0.0}
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
