"""Small-scale 7-env gamut, "new" GR (balanced renormalization + split-moment
Adam) with heavy coherence-with-penalty.

Reproduces matrix_gr_5envs_balanced_splitmoment_shortsteps_nocoh_2seed's config
(balanced renorm + split-moment, MLP m16, classic routing, hack_frac=0.5,
beta=0.05) but turns coherence ON at 512 samples/step with a -2.0 reward penalty
applied to detected hacks DURING the coherence steps, and NO verified-retain
samples. Extended to all 7 small-scale envs.

  - coh_samples_per_rollout = 512  (coherence = same_reward)
  - coherence_rh_mode = penalty, coherence_rh_penalty = 2.0
  - rh_detector_verifies_retain_samples = False  (no verified-retain slice)
  - renormalization_mode = balanced, split_moment = True

Per-env step counts (small_scale_reference): object_qa/persona/sorting=1000,
repeat=500, addition_v2/cities_qa=2000, topic=1000.

GR runs only. 7 envs x 3 seeds = 21 runs.

Launch (all GPUs, 5 concurrent/GPU):
    python -u sweep.py --name <name> --config sweeps/smallscale_newgr_coh512pen2_3seed.py --no_baseline
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

_seeds = [1, 2, 3]

# New GR + heavy coherence-with-penalty, no verified-retain slice.
_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coh_samples_per_rollout": 512,
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 2.0,
    "rh_detector_verifies_retain_samples": False,
}

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_gr_cls_coh512_pen2_noretain_balanced_splitmoment_hf050_st{steps}_s{seed}",
        })

per_gpu = 5
no_baseline = True
