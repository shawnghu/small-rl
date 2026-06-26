"""graft-port FIRST-SLICE smoke: balanced + split_moment + λ=1, classic AND exclusive
(exclusive is the port's new capability — master only stubbed it), MLP m16 (κ=2), one
fast toy env, few steps. Exercises the λ/κ mask cells, the κ geometry guard (passes at
κ=2), SplitMomentAdamW with tagged retain/forget groups, and the per-param capture guard.

Local (1 GPU):  CUDA_VISIBLE_DEVICES=0 python -u sweep.py --name graft_port_smoke --config sweeps/graft_port_smoke.py --no_baseline
Modal:          python -u sweep.py --name graft_port_smoke --config sweeps/graft_port_smoke.py --no_baseline --backend modal
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_ENV = "persona_qa_flattery_conditional_3xreward"            # a fast toy env
_env = next(e for e in _envs if _env_short(e["config"]) == _ENV)

_base = {
    **_shared, **_env,
    "adapter_type": "mlp",
    "mlp_config": "m16",                                     # retain=forget=16 -> κ=(2,2)
    "renormalization_mode": "balanced",
    "split_moment": True,
    "routing_lambda": 1.0,
    "coh_samples_per_rollout": 0,                            # no coherence (core smoke)
    "max_steps": 40,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "seed": 1,
}

runs = [
    {**_base, "routing_mode": "classic",
     "run_name": "graft_port_classic_nocoh_lam1_s1"},
    {**_base, "routing_mode": "exclusive",                   # the port's new exclusive
     "run_name": "graft_port_exclusive_nocoh_lam1_s1"},
]

per_gpu = 2
no_baseline = True
