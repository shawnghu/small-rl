"""Exp 2 — 2-adapter training, linear forget-scale decay, NO coherence.

Train in the 2-adapter config; forget scale follows fs(t)=max(0,1-step/max_steps)
for BOTH generation and the update forward (forget_scale_modulation=linear_decay).
No coherence (coh_samples_per_rollout=0). Two variants:
  - routing ON  = routing_mode=classic (balanced + split_moment): the routed hack
    rep fades from deployment as the scale decays.
  - routing OFF = routing_mode=none (plain 2-adapter forward/backward; both
    adapters live + updated). balanced requires GR, so the none cell uses
    renormalization_mode=off + split_moment=False.

Base = smallscale_warmstart_coh128_lam1_3seed (v2 warm start for sort+topic),
coherence_rh_mode=none (no penalty). cities/addition steps cut to 1000.
M/N = 256/256 (rollout_batch_size=256 routed, coh_samples_per_rollout=256). The
routing-off variant relies on the relaxed coh>0 assert (homogeneous path handles
coherence microbatches; only coherence_rh_mode=none is valid there).

2 routing variants x 7 envs x 3 seeds = 42 runs.
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
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

# Two routing variants. routing=none disables GR, so balanced/split_moment must be
# turned off (plain 2-adapter update).
_variants = [
    {"routing_mode": "classic"},  # inherits balanced + split_moment from _new
    {"routing_mode": "none", "renormalization_mode": "off", "split_moment": False},
]

runs = []
for variant in _variants:
    rtag = variant["routing_mode"]
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_new, **variant,
                "forget_scale_modulation": "linear_decay",
                "coherence": "same_reward",
                "coh_samples_per_rollout": 256,
                "rollout_batch_size": 256,
                "warmstart_data": _warmstart_for(ename),
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp2_lindecay_coh256_route-{rtag}_ws_st{steps}_s{seed}"),
            })

assert len(runs) == len(_variants) * len(_envs) * len(_seeds) == 28, len(runs)

per_gpu = 5
no_baseline = True
