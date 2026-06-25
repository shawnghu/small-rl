"""Reduced-steps rerun of matrix_gr_5envs_balanced_splitmoment_2seed.

Same config as that sweep (balanced renorm + split-moment, MLP m16, classic
routing, interlaced coherence cspr=32, verifier on, hack_frac=0.5, lr=3e-4,
beta=0.05) but with fewer steps per env — the previous run reached a hack
equilibrium quickly, so the long tail was uninformative. Steps:
  object_qa, persona, sorting  -> 1000
  repeat                       -> 500
  addition_v2                  -> 2000 (unchanged; not in the reduction list)

GPU 0 only (GPU 1 in use elsewhere); 5 runs/GPU -> 2 waves of 5.

5 envs x 2 seeds = 10 runs.

Launch (GPU 0 only):
    CUDA_VISIBLE_DEVICES=0 python -u sweep.py --name matrix_gr_5envs_balanced_splitmoment_shortsteps_2seed --config sweeps/matrix_gr_5envs_balanced_splitmoment_shortsteps_2seed.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_KEEP = {
    "object_qa_sycophancy_conditional",
    "sorting_copy_conditional",
    "addition_v2_sycophancy_conditional",
    "repeat_extra_conditional",
    "persona_qa_flattery_conditional_3xreward",
}

# Per-env step counts (override the env defaults from matrix_gr_7envs).
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 2000,  # unchanged (not in the reduction list)
}

_seeds = [1, 2]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
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
            "run_name": f"{ename}_gr_cls_cspr32_balanced_splitmoment_hf050_st{_steps[ename]}_s{seed}",
        })

# GPU 0 only (launch with CUDA_VISIBLE_DEVICES=0); 5 concurrent / GPU.
per_gpu = 5
no_baseline = True
