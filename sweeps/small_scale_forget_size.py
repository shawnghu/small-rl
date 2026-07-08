"""Forget-adapter-size sweep across all 7 small-scale envs.

Same recipe as sweeps/small_scale_reference.py (balanced renorm + split-moment,
classic routing, interlaced coherence cspr=32, verifier on, hack_frac=0.5,
recall=1.0, lr=3e-4, beta=0.05) EXCEPT:
  - retain adapter stays 16 neurons; forget adapter sweeps over {2, 4, 8}
    (presets m16f2 / m16f4 / m16f8).
  - force_kappa=2.0: kappa_r=kappa_f=2.0, pinned to the equal-sized m16
    reference value ((16+16)/16=2). Holds the forget adapter's per-coordinate
    step amplification CONSTANT across sizes (size-derived would give
    kappa_f=9/5/3 for n_F=2/4/8), so the ladder varies only capacity and
    compares cleanly to the m16 balanced recipe. w_floor=2 <= graft_w_max=4, so
    no --graft_w_max bump needed.
  - 3 seeds.

Per-env step counts (from small_scale_reference):
  object_qa, persona, sorting  -> 1000
  repeat                       -> 500
  addition_v2                  -> 2000
  cities_qa                    -> 2000
  topic                        -> 1000

7 envs x 3 forget-sizes x 3 seeds = 63 runs.

Launch (all 8 local H100s + MPS):
    python -u sweep.py --name small_scale_forget_size \
        --config sweeps/small_scale_forget_size.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_KEEP = {
    "object_qa_sycophancy_conditional",
    "sorting_copy_conditional",
    "addition_v2_sycophancy_conditional",
    "repeat_extra_conditional",
    "persona_qa_flattery_conditional_3xreward",
    "cities_qa_sycophancy_conditional",
    "topic_contains_conditional",
}

# Per-env step counts (override matrix_gr_7envs env defaults; match small_scale_reference).
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 2000,
    "cities_qa_sycophancy_conditional": 2000,
    "topic_contains_conditional": 1000,
}

# forget preset -> forget-neuron count (retain stays 16). Tag = forget size.
_forget_presets = {"m16f2": 2, "m16f4": 4, "m16f8": 8}

_seeds = [1, 2, 3]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "force_kappa": 2.0,
}

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    if ename not in _KEEP:
        continue
    for preset, fsize in _forget_presets.items():
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_new,
                "mlp_config": preset,            # overrides _shared's "m16"
                "max_steps": _steps[ename],
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (f"{ename}_gr_cls_cspr32_balanced_splitmoment_"
                             f"m16f{fsize}_k2_hf050_st{_steps[ename]}_s{seed}"),
            })

# per_gpu=5: intended concurrency for a healthy H100/H200 pod (small-scale 135M).
# RAM CAVEAT: needs ~2-3GB NORMAL host RAM per run. If a pod strands most RAM in
# reserved hugepages (check `grep HugePages_Total /proc/meminfo`; `free` inside a
# container shows the physical host incl. hugepages, so it can look ample while
# MemAvailable is ~0), the vLLM inits get OOM-killed — drop per_gpu (1-2) or use a
# pod without the reservation. GPU compute is not the bottleneck at this scale.
per_gpu = 5
no_baseline = True
