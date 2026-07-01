"""Exp 4 — GR as a retain-hack PROHIBITION (not representation learning).

128 reinterpreted "routing" samples (retain advantage multiplier -1, raw reward,
NO penalty) + 512 STANDARD coherence (coherence_rh_mode=none). Implemented as a
direct override of the four routing grad-mask constants + a routing forward
forget-scale (--retain_prohibition_mode {a,b,c}), with split_moment OFF (plain
Adam => the literal forget multiplier is an accumulation weight, no kappa/clamp).
  (a) all routing (1,-1,1); (b) off-policy retain-only update (0,-1,0), old_logps
  at (1,1); (c) good (1,-1,1) / bad (1,-1,3).

Base = smallscale_warmstart_coh128_lam1_3seed (v2 warm start for sort+topic),
coherence_rh_mode=none. cities/addition steps cut to 1000.
M/N = 128/512 -> rollout_batch_size=128, coh_samples_per_rollout=512, total 640.

3 modes x 7 envs x 3 seeds = 63 runs.
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2]

# balanced (keeps the routing-group advantage) but split_moment OFF (required by
# retain_prohibition: literal plain-Adam multiplier semantics).
_new = {
    "renormalization_mode": "balanced",
    "split_moment": False,
    "coherence": "same_reward",
    "coherence_rh_mode": "none",
    "rh_detector_verifies_retain_samples": False,
}

_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,
    "topic_contains_conditional": 1000,
}

_V2_ENVS = {"sorting_copy_conditional", "topic_contains_conditional"}
def _warmstart_for(ename):
    return "warmstart_data_v2" if ename in _V2_ENVS else "warmstart_data"

_modes = ["a", "b", "c"]

runs = []
for mode in _modes:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_new,
                "retain_prohibition_mode": mode,
                "rollout_batch_size": 128,
                "coh_samples_per_rollout": 512,
                "warmstart_data": _warmstart_for(ename),
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp4_retainprohib_{mode}_ws_st{steps}_s{seed}"),
            })

assert len(runs) == len(_modes) * len(_envs) * len(_seeds) == 42, len(runs)

per_gpu = 5
no_baseline = True
