"""GRAFT canonical 7-env REGRESSION re-run on the POST-MERGE (graft-port) pipeline.

Reproduces the graft-routing `sweeps/graft_canonical_7envs.py` GRAFT arm — classic
routing, λ=1, NO coherence — but selects the method the graft-port way:
`renormalization_mode=balanced` + `split_moment=True` (the fused single-backward
decouple). On graft-routing the same method was the multi-pass GraftAdam; the only
expected behavioral difference at this config is the KL handling (whole-loss-masked
here vs the old 3-pass KL-faithful), which should be minimal. κ=2 auto from m16.

NO-INTERVENTION is NOT re-run: routing_mode=none takes the standard TRL path,
untouched by the graft-port changes, so the existing
output/graft_canon_7envs_fseval/*_noint_lam1_s*.json baseline is reused verbatim.

Run names are IDENTICAL to the old graft arm (`<env>_graft_lam1_s<seed>`) so the
forget-scale eval / collate / pareto (run from graft-routing, inference is
branch-identical) line up env+seed for a direct regression comparison.

7 envs × 3 seeds = 21 runs.

Launch (graft-port):
    python -u sweep.py --name graft_canon_port --config sweeps/graft_canonical_7envs_port.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.no_intervention_7envs import _base as _noint_base, _envs, _env_short

# graft = the canonical no-intervention base + classic routing λ=1, selected via the
# graft-port balanced/split-moment fused kernel.
_GRAFT = {
    **_noint_base,
    "routing_mode": "classic",
    "routing_lambda": 1.0,
    "renormalization_mode": "balanced",
    "split_moment": True,
}

_SEEDS = [1, 2, 3]

runs = []
for cfg, max_steps, extras in _envs:
    ename = _env_short(cfg)
    for s in _SEEDS:
        runs.append({
            **_GRAFT,
            **extras,
            "config": cfg,
            "max_steps": max_steps,     # per-env: 2000 (most) / 1000 (repeat, topic)
            "eval_every": 0,            # post-hoc forget-scale eval only (matches old)
            "seed": s,
            "run_name": f"{ename}_graft_lam1_s{s}",
        })

no_baseline = True
pack_runs = False    # 1 run / container (avoids the Modal vLLM-init race; --no_pack)
per_gpu = 1
