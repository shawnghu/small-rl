"""GRAFT plumbing smoke test — 1 run, 3 steps, classic routing + interlaced coherence.

Validates the train.py multi-pass surgery end-to-end on Modal (no local GPU): does the
GRAFT path (compute_advantages -> _graft_forward_backward 3-pass routing / 1-pass coherence
-> GraftAdam window step) actually run without crashing, with gradients flowing to both
adapters. Reuses the known-good binary_dynamics_persona classic+coherence base; only smoke
overrides differ. eval off + no save to isolate the training/optimizer path.

Launch:
    python -u sweep.py --name graft_smoke --config sweeps/graft_smoke.py --backend modal \\
        --no_baseline --no_filter_baseline --no_reward_penalty_baseline
"""
from sweeps.binary_dynamics_persona_1000 import _base

runs = [{
    **_base,
    "max_steps": 3,
    "eval_every": 0,        # eval off for the smoke (isolate the training path)
    "save_steps": 999,      # no checkpoint write
    "seed": 1,
    "routing_lambda": 1.0,  # clean routing
    "run_name": "graft_smoke_persona_cls_coh_lam1_s1",
}]

no_baseline = True
