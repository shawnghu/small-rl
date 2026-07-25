"""Topic 5C (prefix-monitored) pilot: canonical no-coherence GRAFT only, 3 seeds.

First training contact for the reviewer-suggested eval-awareness-prefix topic
variant (configs/test_new_envs/topic_contains_monitored.yaml; see envs/topic.py
sub_env 5C and the 2026-07-24 step-0 calibration in
output/topic_step0_prefix/). No baselines yet ("see if this works at all" —
Jake): donothing/RP arms come later if routing separates.

Method matches sweeps/graft_canonical_7envs_port.py exactly (classic routing,
λ=1, balanced renorm + split-moment, NO coherence, eval_every=0 with post-hoc
forget-scale eval), so results are protocol-comparable to the 5A topic arm
(output/graft_canon_port_fseval/topic_contains_conditional_graft_lam1_s*).

3 runs. Launch (Modal, 1 run/container — no MPS on Modal):
    .venv/bin/python -u sweep.py --name topic5c_graft_nocoh \
        --config sweeps/topic5c_graft_nocoh_3seed.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.no_intervention_7envs import _base as _noint_base

_GRAFT = {
    **_noint_base,
    "routing_mode": "classic",
    "routing_lambda": 1.0,
    "renormalization_mode": "balanced",
    "split_moment": True,
}

_SEEDS = [1, 2, 3]

runs = [
    {
        **_GRAFT,
        "config": "configs/test_new_envs/topic_contains_monitored.yaml",
        "max_steps": 1000,
        "eval_every": 0,   # post-hoc forget-scale eval only (canonical protocol)
        "seed": s,
        "run_name": f"topic_contains_monitored_graft_lam1_s{s}",
    }
    for s in _SEEDS
]

no_baseline = True
pack_runs = False    # 1 run / container
per_gpu = 1
