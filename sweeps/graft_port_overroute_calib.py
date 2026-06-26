"""graft-port slice-2b over-routing CALIBRATION: demonstrate λ>1 TRAINS and
measure the realized per-coordinate Adam step vs λ.

The λ=1.5 cells of graft_port_slow_smoke correctly tripped the realized-step gate
at ~82× lr (W_MAX=4): over-routing's masked-m / natural-v per-coordinate ratio is
NOT bounded by the mask-weight cap (the plan's step_bound.py predicted this). Here
we run MILDER over-routing (λ∈{1.1,1.25}) with a generous graft_w_max so the gate
does not trip, to (a) prove the over-routing path produces a real training run and
(b) read graft/realized_step_{p999,max} to calibrate the realized step vs λ.

Modal:  python -u sweep.py --name graft_port_overroute_calib --config sweeps/graft_port_overroute_calib.py --no_baseline --backend modal --no_pack
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_ENV = "persona_qa_flattery_conditional_3xreward"
_env = next(e for e in _envs if _env_short(e["config"]) == _ENV)

_base = {
    **_shared, **_env,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": False,
    "graft_w_max": 64.0,                 # generous so the gate measures, not rejects
    "max_steps": 40,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "seed": 1,
}

runs = [
    {**_base, "routing_mode": m, "routing_lambda": lam,
     "run_name": f"overroute_{m}_lam{str(lam).replace('.','p')}_s1"}
    for m in ("classic", "exclusive")
    for lam in (1.1, 1.25)
]

per_gpu = 4
no_baseline = True
