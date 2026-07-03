"""Exp 3 n1_off at a 128/512 coherence-heavy mix, on object/persona/repeat/topic,
3 seeds each.

Same n1_off cell config (coh_forget_scale=-1, coh_forget_grad=off,
routing_forget_scale=1, balanced + split_moment, coherence same_reward,
coherence_rh_mode=none, warm start, v2 data for topic) EXCEPT the routing/coherence
balance is M/N = 128/512 (rollout_batch_size=128 routing, coh_samples_per_rollout=512
coherence; total 640) instead of 256/256, and envs limited to
{object_qa, persona, repeat, topic}.

4 envs x 3 seeds = 12 runs. GR runs only.
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
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "repeat_extra_conditional": 500,
    "topic_contains_conditional": 1000,
}

_TARGET = set(_steps)
_V2_ENVS = {"sorting_copy_conditional", "topic_contains_conditional"}
def _warmstart_for(ename):
    return "warmstart_data_v2" if ename in _V2_ENVS else "warmstart_data"

_envs4 = [e for e in _envs if _env_short(e["config"]) in _TARGET]
assert len(_envs4) == 4, [_env_short(e["config"]) for e in _envs4]

# M/N = 128/512 (both multiples of num_generations=32; total 640).
assert 128 % 32 == 0 and 512 % 32 == 0

runs = []
for env in _envs4:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "coh_forget_scale": -1.0,
            "coh_forget_grad": "off",
            "routing_forget_scale": 1.0,
            "rollout_batch_size": 128,
            "coh_samples_per_rollout": 512,
            "warmstart_data": _warmstart_for(ename),
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_exp3_n1off_M128N512_3seed_ws_st{steps}_s{seed}"),
        })

assert len(runs) == len(_envs4) * len(_seeds) == 12, len(runs)

per_gpu = 5
no_baseline = True
