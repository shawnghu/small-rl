"""Validation run for the unified diagnostic channels (post-refactor defaults).

Same 5-env GR matrix cell as sweeps/matrix_gr_5envs_graddiag.py (classic
routing, interlaced coherence cspr=32, mlp m16, verifier on, lr=3e-4, beta=0.05,
hack_frac=0.5, recall=1.0), but:
  - NO weight decay (base _shared; the wd1 variant added it)
  - DEFAULT diagnostic levels: routing-trace and adapter-diagnostics both at
    their defaults (when_eval interval, adapter_diag_level=adapter_diagnostics).
    The old per-sample grad recompute (grad_diag_every) is intentionally NOT
    enabled — this exercises the new default-on channels end-to-end.
  - 2 seeds per env.

5 envs x 2 seeds = 10 runs.

Launch (local backend, 2xH200, MPS managed by sweep.py):
    python -u sweep.py --name matrix_gr_5envs_defaultdiag_2seed \\
        --config sweeps/matrix_gr_5envs_defaultdiag_2seed.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_KEEP = {
    "object_qa_sycophancy_conditional",
    "sorting_copy_conditional",
    "addition_v2_sycophancy_conditional",
    "repeat_extra_conditional",
    "persona_qa_flattery_conditional_3xreward",
}

_seeds = [1, 2]

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
            "seed": seed,
            "run_name": f"{ename}_gr_cls_cspr32_hf050_rcl100_defaultdiag_s{seed}",
        })

# Local backend: 2 GPUs, MPS-shared. 5/GPU -> all 10 runs concurrent.
per_gpu = 5
no_baseline = True
