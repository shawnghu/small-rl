"""Smoke/validation rerun of matrix_gr_5envs_defaultdiag_2seed with the new
'balanced' renormalization + split-moment optimizer.

Identical to sweeps/matrix_gr_5envs_defaultdiag_2seed.py — same 5 envs, 2 seeds,
SmolLM2-135M, classic routing, MLP m16, interlaced coherence cspr=32, verifier on,
lr=3e-4, beta=0.05, hack_frac=0.5, recall=1.0, default diagnostics — except the
two new advantage/optimizer settings:

  - renormalization_mode = "balanced"  (#1 clean baseline on routing groups + #2
                                         forget redistribution in the fused kernel)
  - split_moment = True                (#3 Adam v <- pre-routing grad, m <- routed grad)

balanced/split now compose with MLP adapters, interlaced coherence, and the
verifier (coherence groups are handled as before; #1/#2 touch routing groups only;
coherence passes feed both Adam moments with weight 1), so the original cell's
setup is preserved unchanged apart from these two flags.

5 envs x 2 seeds = 10 runs.

Launch (local backend, 2xH200, MPS managed by sweep.py):
    python -u sweep.py --name matrix_gr_5envs_balanced_splitmoment_2seed --config sweeps/matrix_gr_5envs_balanced_splitmoment_2seed.py --no_baseline
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
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_gr_cls_cspr32_balanced_splitmoment_hf050_rcl100_s{seed}",
        })

# Local backend: 2 GPUs, MPS-shared. 5/GPU -> all 10 runs concurrent.
per_gpu = 5
no_baseline = True
