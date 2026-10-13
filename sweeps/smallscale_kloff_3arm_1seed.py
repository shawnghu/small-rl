"""Small-scale 7-env gamut, KL penalty OFF (beta=0). Three arms per env, 1 seed
each: GR, RP, and do-nothing baseline.

GR here is the "old" GR (pre-b81c12): balanced renormalization OFF (historical
retain-only default) and NO split-moment Adam. Otherwise canonical small-scale
config (== canonical_topups_4envs but beta=0 and extended to all 7 envs):
classic routing + interlaced coherence cspr=32 (penalty mode, verifier on),
MLP m16, lr=3e-4, hack_frac=0.5, rh_detector_recall=1.0.

  - GR          : routing_mode=classic, renorm=retain-only, split_moment=False
  - RP          : routing_mode=none, reward_penalty_baseline, pen=2.0 (canonical)
  - do-nothing  : routing_mode=none, coherence stripped (vanilla RL)

The only deliberate deviation from canonical is beta=0 (KL off) on every arm.

Per-env step counts (small_scale_reference): object_qa/persona/sorting=1000,
repeat=500, addition_v2/cities_qa=2000, topic=1000.

7 envs x 3 arms x 1 seed = 21 runs.

Launch (all GPUs, 5 concurrent/GPU):
    python -u sweep.py --name <name> --config sweeps/smallscale_kloff_3arm_1seed.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

# Per-env step counts (override env defaults from matrix_gr_7envs).
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 2000,
    "cities_qa_sycophancy_conditional": 2000,
    "topic_contains_conditional": 1000,
}

_seed = 1

# KL off on every arm; canonical hack_frac / recall.
_common = {
    "beta": 0,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}

# Old GR: balanced renorm OFF (historical retain-only default), no split-moment.
_OLD_GR = {"renormalization_mode": "retain-only", "split_moment": False}

# Keys carrying coherence/routing config that the do-nothing baseline must drop
# (mirrors sweep.py's ROUTING_ONLY_PARAMS strip so vanilla RL asserts hold:
# coh_samples_per_rollout>0 requires routing_mode!=none or reward_penalty).
_DONOTHING_STRIP = {
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
}

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    base = {**_shared, **env, **_common, "max_steps": steps, "seed": _seed}

    # GR arm: classic routing + coherence cspr=32 (from _shared), old-GR renorm.
    runs.append({
        **base, **_OLD_GR,
        "routing_mode": "classic",
        "run_name": f"{ename}_gr_cls_cspr32_kloff_oldgr_hf050_st{steps}_s{_seed}",
    })

    # RP arm: reward-penalty baseline (pen=2.0), verified-retain extras kept
    # (coh=32 + verifier on, from _shared) — exactly canonical_topups RP.
    runs.append({
        **base,
        "routing_mode": "none",
        "reward_penalty_baseline": True,
        "reward_penalty_amount": 2.0,
        "rp_extra_retain_advantage_multiplier": 1.0,
        "run_name": f"{ename}_rp_pen2_kloff_hf050_st{steps}_s{_seed}",
    })

    # Do-nothing baseline: vanilla RL (no routing, no coherence).
    dn = {k: v for k, v in base.items() if k not in _DONOTHING_STRIP}
    dn["routing_mode"] = "none"
    dn["run_name"] = f"{ename}_donothing_kloff_hf050_st{steps}_s{_seed}"
    runs.append(dn)

per_gpu = 5
no_baseline = True
