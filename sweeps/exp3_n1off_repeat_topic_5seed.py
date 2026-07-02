"""Exp 3 consistency re-run: cell n1_off (coh_forget_scale=-1, coh_forget_grad=off,
routing_forget_scale=1) on repeat + topic only, 5 seeds — to check the n1_off
result is consistent on those two envs (was only 2 seeds before).

EXACT same config as the exp3 n1_off cell in sweeps/exp3_neg_deploy.py (256/256,
balanced + split_moment, coherence same_reward, coherence_rh_mode=none, warm start,
v2 data for topic). Only difference: envs limited to {repeat, topic} and 5 seeds.

2 envs x 5 seeds = 10 runs. GR runs only.
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2, 3, 4, 5]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coherence_rh_mode": "none",
    "rh_detector_verifies_retain_samples": False,
}

_steps = {
    "repeat_extra_conditional": 500,
    "topic_contains_conditional": 1000,
}

_TARGET = set(_steps)              # repeat + topic
_V2_ENVS = {"sorting_copy_conditional", "topic_contains_conditional"}
def _warmstart_for(ename):
    return "warmstart_data_v2" if ename in _V2_ENVS else "warmstart_data"

_envs2 = [e for e in _envs if _env_short(e["config"]) in _TARGET]
assert len(_envs2) == 2, [_env_short(e["config"]) for e in _envs2]

runs = []
for env in _envs2:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "coh_forget_scale": -1.0,
            "coh_forget_grad": "off",
            "routing_forget_scale": 1.0,
            "rollout_batch_size": 256,
            "coh_samples_per_rollout": 256,
            "warmstart_data": _warmstart_for(ename),
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_exp3_n1off_5seed_ws_st{steps}_s{seed}"),
        })

assert len(runs) == len(_envs2) * len(_seeds) == 10, len(runs)

per_gpu = 5
no_baseline = True
