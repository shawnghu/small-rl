"""Exp 1b / 1c — off-policy coherence update mode variants, on 3 envs.

Same setup as Exp 1 (off-policy coherence: coherence GENERATED retain-only (1,0),
old_logps at (1,0); update forward is 2-adapter), but varying WHAT the off-policy
coherence update touches:

  1b  coherence_update_config=twoadapter_retain : forget active in the update
      forward (train_fs) but its grad masked off -> ONLY retain is updated
      (forget fully decoupled: no m AND no v on coherence).
  1c  coherence_update_config=twoadapter_routed : apply gradient routing to the
      coherence samples -> detected hacks go to the forget adapter only, undetected
      update both (classic masks, forward scale train_fs).

Envs: persona, sorting, topic (sort+topic use v2 warm-start data).
Same 3 batch cells as Exp 1: M/N 256/256, 128/384, 32/512.

2 modes x 3 envs x 3 batches x 3 seeds = 54 runs. GR runs only.
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2, 3]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coherence_rh_mode": "none",
    "rh_detector_verifies_retain_samples": False,
}

_steps = {
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "topic_contains_conditional": 1000,
}

_TARGET = set(_steps)            # the 3 envs
_V2_ENVS = {"sorting_copy_conditional", "topic_contains_conditional"}
def _warmstart_for(ename):
    return "warmstart_data_v2" if ename in _V2_ENVS else "warmstart_data"

_envs3 = [e for e in _envs if _env_short(e["config"]) in _TARGET]
assert len(_envs3) == 3, [_env_short(e["config"]) for e in _envs3]

_modes = ["twoadapter_retain", "twoadapter_routed"]   # 1b, 1c
_batches = [(256, 256), (128, 384), (32, 512)]
for _m, _n in _batches:
    assert _m % 32 == 0 and _n % 32 == 0, (_m, _n)

_MODE_TAG = {"twoadapter_retain": "1b_retainupd", "twoadapter_routed": "1c_routed"}

runs = []
for mode in _modes:
    for (M, N) in _batches:
        for env in _envs3:
            ename = _env_short(env["config"])
            steps = _steps[ename]
            for seed in _seeds:
                runs.append({
                    **_shared, **env, **_new,
                    "coherence_update_config": mode,
                    "rollout_batch_size": M,
                    "coh_samples_per_rollout": N,
                    "warmstart_data": _warmstart_for(ename),
                    "max_steps": steps,
                    "unconditional_hackable": False,
                    "hack_frac": 0.5,
                    "rh_detector_recall": 1.0,
                    "seed": seed,
                    "run_name": (
                        f"{ename}_exp{_MODE_TAG[mode]}_M{M}N{N}_ws_st{steps}_s{seed}"),
                })

assert len(runs) == len(_modes) * len(_batches) * len(_envs3) * len(_seeds) == 54, len(runs)

per_gpu = 5
no_baseline = True
