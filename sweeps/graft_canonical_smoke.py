"""2-run smoke for the GRAFT canonical experiment: validates the NO-COHERENCE GRAFT path
(cspr=0) + the no-intervention companion before the full sweep. persona env, 4 steps.

Launch (warm cache first if FS flaky):
    python -u sweep.py --name graft_canon_smoke --config sweeps/graft_canonical_smoke.py \\
        --backend modal --no_baseline --no_filter_baseline --no_reward_penalty_baseline
"""
from sweeps.graft_canonical_7envs import _GRAFT, _NOINT

_Y = "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml"

runs = [
    {**base, "config": _Y, "max_steps": 4, "save_steps": 999, "eval_every": 0,
     "seed": 1, "run_name": f"graft_canon_smoke_persona_{cond}_s1"}
    for cond, base in (("graft", _GRAFT), ("noint", _NOINT))
]

no_baseline = True
pack_runs = True
per_gpu = 1
