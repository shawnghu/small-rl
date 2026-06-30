"""Exp 5 — large retain-only weight decay + ema_clamp forget-scale controller.

Standard GR (classic routing + coherence, coherence_rh_mode=none) with a large
weight decay on the RETAIN adapter only (forget never decays — see the worktree's
_build_retain_forget_groups), to passively destroy the retain hack rep, plus the
existing ema_clamp controller attenuating the forget scale to a target hack rate
of 0.3 (forget_scale_modulation=ema_clamp, forget_scale_target_hack_rate=0.3).

Base = smallscale_warmstart_coh128_lam1_3seed (v2 warm start for sort+topic),
coherence_rh_mode=none. cities/addition steps cut to 1000. M/N = 512/128 (base).

weight_decay ladder starts at 1.0 (researcher expects to go higher).
3 wd x 7 envs x 3 seeds = 63 runs.
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
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,
    "topic_contains_conditional": 1000,
}

_V2_ENVS = {"sorting_copy_conditional", "topic_contains_conditional"}
def _warmstart_for(ename):
    return "warmstart_data_v2" if ename in _V2_ENVS else "warmstart_data"

# Retain-only weight-decay ladder; start at 1.0 (extendable upward).
_weight_decays = [1.0, 3.0, 10.0]

runs = []
for wd in _weight_decays:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            wdtag = f"{wd:g}".replace(".", "p")
            runs.append({
                **_shared, **env, **_new,
                "weight_decay": wd,
                "forget_scale_modulation": "ema_clamp",
                "forget_scale_target_hack_rate": 0.3,
                "coh_samples_per_rollout": 128,
                "rollout_batch_size": 512,
                "warmstart_data": _warmstart_for(ename),
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp5_wd{wdtag}_emaclamp03_ws_st{steps}_s{seed}"),
            })

assert len(runs) == len(_weight_decays) * len(_envs) * len(_seeds) == 63, len(runs)

per_gpu = 5
no_baseline = True
