"""No-coherence A/B partner to matrix_gr_5envs_balanced_splitmoment_shortsteps_2seed.

Identical to that sweep (balanced renorm + split-moment, MLP m16, classic routing,
hack_frac=0.5, same reduced per-env steps: obj/persona/sort=1000, repeat=500,
addition=2000) but with NO coherence training at all:
  - coherence = "none", coh_samples_per_rollout = 0
  - rh_detector_verifies_retain_samples = False  (the verifier operates on the
    coherence slice; with cspr=0 it must be off — train.py asserts cspr>0 otherwise)

Purpose: A/B against the coherence run — does coherence training help the retain
adapter recover after the hack equilibrium? With no coherence, balanced's #1 clean
baseline applies to every (routing) group; #2 redistribution and split-moment are
unchanged.

GPU 1 only (launched once GPU 1 frees); 5 runs/GPU -> 2 waves of 5.

5 envs x 2 seeds = 10 runs.

Launch (GPU 1 only):
    CUDA_VISIBLE_DEVICES=1 python -u sweep.py --name matrix_gr_5envs_balanced_splitmoment_shortsteps_nocoh_2seed --config sweeps/matrix_gr_5envs_balanced_splitmoment_shortsteps_nocoh_2seed.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_KEEP = {
    "object_qa_sycophancy_conditional",
    "sorting_copy_conditional",
    "addition_v2_sycophancy_conditional",
    "repeat_extra_conditional",
    "persona_qa_flattery_conditional_3xreward",
}

_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 2000,
}

_seeds = [1, 2]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    # No coherence training (and therefore no retain verifier).
    "coherence": "none",
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": False,
}

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    if ename not in _KEEP:
        continue
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "max_steps": _steps[ename],
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_gr_cls_nocoh_balanced_splitmoment_hf050_st{_steps[ename]}_s{seed}",
        })

# GPU 1 only (launch with CUDA_VISIBLE_DEVICES=1); 5 concurrent / GPU.
per_gpu = 5
no_baseline = True
