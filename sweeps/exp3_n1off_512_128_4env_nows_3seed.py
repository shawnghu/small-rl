"""Exp 3 n1_off at a 512/128 (routing-heavy) mix — the flip of the 128/512 sweep —
on object/persona/repeat/topic, 3 seeds, WITHOUT warm-start SFT.

Same as sweeps/exp3_n1off_128_512_4env_nows_3seed.py EXCEPT M/N = 512/128
(rollout_batch_size=512 routing, coh_samples_per_rollout=128 coherence; total 640)
instead of 128/512. No warmstart_data -> RL from base model + zero-init adapters.

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
_envs4 = [e for e in _envs if _env_short(e["config"]) in _TARGET]
assert len(_envs4) == 4, [_env_short(e["config"]) for e in _envs4]
assert 512 % 32 == 0 and 128 % 32 == 0

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
            "rollout_batch_size": 512,
            "coh_samples_per_rollout": 128,
            # NO warmstart_data: skip warm-start SFT pre-localization.
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_exp3_n1off_M512N128_nows_3seed_st{steps}_s{seed}"),
        })

assert len(runs) == len(_envs4) * len(_seeds) == 12, len(runs)

per_gpu = 5
no_baseline = True
