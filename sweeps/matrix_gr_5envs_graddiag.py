"""matrix_gr_7envs reproduction, restricted + per-sample grad/activation diagnostic.

Caveats vs sweeps/matrix_gr_7envs.py:
  - the canonical (hack_frac=0.5, rh_detector_recall=1.0) cell only
    (that sweep deliberately *skips* this cell; here it's exactly what we run)
  - 5 envs: object_qa, sorting, addition, repeat, persona_qa
  - 3 seeds (1,2,3)
  - --grad_diag_every on (gradient + activation 2x2 per-sample diagnostic)

Same GR config otherwise (_shared / _envs imported verbatim): classic routing,
interlaced coherence cspr=32, mlp m16, verifier on, lr=3e-4, beta=0.05.

5 envs x 3 seeds = 15 runs.

Goal: look for a reliable per-sample signal (gradient or activation norm) that
distinguishes forget (hack) samples from retain samples — e.g. the forget
adapter's activation being disproportionately large on forget samples.

Launch:
    python -u sweep.py --name matrix_gr_5envs_graddiag \\
        --config sweeps/matrix_gr_5envs_graddiag.py --backend modal --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_KEEP = {
    "object_qa_sycophancy_conditional",
    "sorting_copy_conditional",
    "addition_v2_sycophancy_conditional",
    "repeat_extra_conditional",
    "persona_qa_flattery_conditional_3xreward",
}

_seeds = [1, 2, 3]
GRAD_DIAG_EVERY = 25

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    if ename not in _KEEP:
        continue
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "grad_diag_every": GRAD_DIAG_EVERY,
            "seed": seed,
            "run_name": f"{ename}_gr_cls_cspr32_hf050_rcl100_graddiag_s{seed}",
        })

# Local backend (sweep.py default): 3 GPUs, MPS-shared. 5/GPU -> all 15 runs
# concurrent. Same vLLM-memory defaults that ran the original matrix_gr_7envs.
per_gpu = 5
no_baseline = True
