"""Exp 1 — off-policy coherence update in the 2-adapter config.

Coherence is GENERATED retain-only (1,0) with old_logps at (1,0) (unchanged);
the UPDATE forward runs the 2-adapter config and BOTH adapters get gradient
(--coherence_update_config twoadapter). See EXPERIMENTS_HACK_SUPPRESSION.md Exp 1.

Base = smallscale_warmstart_coh128_lam1_3seed with the v2 warm-start data for
sort+topic, minus the reward penalty (coherence_rh_mode=none). cities/addition
steps cut to 1000 (warm start -> faster).

Batches (M/N = routing/coherence): rollout_batch_size = M, coh_samples_per_rollout
= N, total = M+N (train.py:5716, dynamic token batching). Cells:
  256/256, 128/384, 32/512.

3 batch cells x 7 envs x 3 seeds = 63 runs. GR runs only.
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2, 3]

# Canonical "new GR" base (smallscale_newgr_coh512pen2_3seed._new) MINUS the
# penalty: coherence_rh_mode=none (no penalty/filter — isolates the intervention).
_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coherence_rh_mode": "none",
    "rh_detector_verifies_retain_samples": False,
}

# Per-env steps (small_scale_reference) with cities/addition cut 2000 -> 1000.
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,
    "topic_contains_conditional": 1000,
}

# v2 warm-start data for sort+topic only; default warmstart_data for the rest.
_V2_ENVS = {"sorting_copy_conditional", "topic_contains_conditional"}
def _warmstart_for(ename):
    return "warmstart_data_v2" if ename in _V2_ENVS else "warmstart_data"

# (M routing, N coherence) cells.
_batches = [(256, 256), (128, 384), (32, 512)]
for _m, _n in _batches:
    assert _m % 32 == 0 and _n % 32 == 0, (_m, _n)

runs = []
for (M, N) in _batches:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_new,
                "coherence_update_config": "twoadapter",
                "rollout_batch_size": M,
                "coh_samples_per_rollout": N,
                "warmstart_data": _warmstart_for(ename),
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp1_offpolicycoh_M{M}N{N}_ws_st{steps}_s{seed}"),
            })

assert len(runs) == len(_batches) * len(_envs) * len(_seeds) == 63, len(runs)

per_gpu = 5
no_baseline = True
