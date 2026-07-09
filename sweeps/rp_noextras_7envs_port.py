"""Reward-penalty baseline with ZERO use of the high-confidence retain
classifier, for the v4 camera-ready figures (2026-07-07).

Every cached RP variant (rp_baseline_*/rp_pen_mult_redo/ratio sweeps) trained
with verified-retain extras (`rh_detector_verifies_retain_samples=True` +
coh_samples_per_rollout=32/128) — i.e. they consumed the high-precision retain
classifier we no longer allow the method (or baselines) to use. The only
no-extras RP data (rp_noextras_3envs_{investigation,extended}) covers just
cities/persona/sorting at pen=2 on the old stack.

This sweep: vanilla RP on all 7 canonical envs, penalty in {2, 5, 10} (the
only surviving best-RP spoke — mult/ratio were properties of the extras),
mirroring the no_intervention_7envs base exactly (same envs/steps/lr/beta/
seeds as the graft_canon_port / coh32 arms; routing_mode=none -> standard TRL
path, no renormalization machinery). No extras, no verified-retain, detector
used ONLY for the penalty.

7 envs x 3 pens x 3 seeds = 63 runs.

Launch (Modal, same shape as the other port sweeps):
    python -u sweep.py --name rp_noextras_7envs_port --config sweeps/rp_noextras_7envs_port.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.no_intervention_7envs import _base as _noint_base, _envs, _env_short

_PENS = [2.0, 5.0, 10.0]
_SEEDS = [1, 2, 3]

runs = []
for cfg, max_steps, extras in _envs:
    ename = _env_short(cfg)
    for pen in _PENS:
        for s in _SEEDS:
            runs.append({
                **_noint_base,
                **extras,
                "config": cfg,
                "max_steps": max_steps,     # per-env: 2000 (most) / 1000 (repeat, topic)
                "reward_penalty_baseline": True,
                "reward_penalty_amount": pen,
                "eval_every": 50,           # piggybacked in-training curves (near-free)
                "seed": s,
                "run_name": f"{ename}_rp_noextras_pen{pen:g}_s{s}",
            })

no_baseline = True
pack_runs = False    # 1 run / container (avoids the Modal vLLM-init race; --no_pack)
per_gpu = 1
