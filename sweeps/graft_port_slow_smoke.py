"""graft-port SLICE-2 smoke: the λ≠1 two-backward v=a_v slow path.

Exercises, on one fast toy env (SmolLM2-135M, MLP m16 → κ=2), a few steps each:
  - λ=1.0 classic   — PARITY: single-backward fast path (= graft_port_smoke), the
    regression anchor (the split of _packed_compute_loss must not change λ=1).
  - λ=0.5 classic   — slice 2a soft routing (2-backward, scalar masks).
  - λ=0.5 exclusive — slice 2a, bidirectional masks.
  - λ=1.1 classic   — slice 2b over-routing (per-group λ_eff cap + B1 v-floor +
    realized-step gate), graft_w_max raised to 64 (the over-routing per-coordinate
    realized step is NOT bounded by the mask-weight cap — at this env's detection
    stats λ=1.5 @ W_MAX=4 correctly TRIPS the gate at ~82× lr; λ=1.1 trains within
    a w_max=64 budget). Watch graft/realized_step_{p999,max}.
  - λ=1.1 exclusive — slice 2b, both adapters absorb.

Validates what CPU can't: liger called TWICE on one shared forward (the m/v
backwards), the per-microbatch .grad snapshot/restore, the accumulator rearm /
double-flush on the real packed path, and that nothing crashes / curves are sane.
(λ=1.5 over-budget gate-fire is a documented + CPU-tested behavior — see
sweeps/graft_port_overroute_calib.py + tests/test_graft_overrouting.py.)

Local (1 GPU):  CUDA_VISIBLE_DEVICES=0 python -u sweep.py --name graft_port_slow_smoke --config sweeps/graft_port_slow_smoke.py --no_baseline
Modal:          python -u sweep.py --name graft_port_slow_smoke --config sweeps/graft_port_slow_smoke.py --no_baseline --backend modal
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
    "coh_samples_per_rollout": 0,                            # no coherence (core slow-path smoke)
    "rh_detector_verifies_retain_samples": False,
    "max_steps": 40,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "seed": 1,
}

runs = [
    {**_base, "routing_mode": "classic",   "routing_lambda": 1.0,
     "run_name": "slow_classic_lam1p0_s1"},                 # parity anchor (fast path)
    {**_base, "routing_mode": "classic",   "routing_lambda": 0.5,
     "run_name": "slow_classic_lam0p5_s1"},                 # 2a soft
    {**_base, "routing_mode": "exclusive", "routing_lambda": 0.5,
     "run_name": "slow_exclusive_lam0p5_s1"},               # 2a soft, bidir
    # The REGRESSION cells: λ=1.5 @ W_MAX=4 tripped the realized-step gate at 82× lr
    # before the per-coordinate clamp. Under the default step_policy='clamp' they now
    # TRAIN — the runaway coords are bounded to ≤4× lr. Watch graft/frac_coords_clamped
    # (>0, the clamp engaging) + graft/lam_eff_mean (the per-group cap) on wandb.
    {**_base, "routing_mode": "classic",   "routing_lambda": 1.5,
     "run_name": "slow_classic_lam1p5_clamp_s1"},
    {**_base, "routing_mode": "exclusive", "routing_lambda": 1.5,
     "run_name": "slow_exclusive_lam1p5_clamp_s1"},
]

per_gpu = 5
no_baseline = True
