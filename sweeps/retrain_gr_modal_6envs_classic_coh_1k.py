"""GR retrain — classic routing + coherence enabled (canonical regime).

Companion to the exclusive + no-coh pilots. Restores the canonical interlaced
coherence path (coh_samples_per_rollout=32, coherence_rh_penalty=3.0,
rh_detector_verifies_retain_samples=True) on top of classic routing. Keeps
max_steps=1000 to match the prior pilots so the comparison is one-variable.

6 envs (topic_contains skipped per user request) x 2 seeds = 12 runs, all on
Modal H100s, single wave. Wall ETA ~85 min (bottlenecked by repeat at
~70 min/run no-coh; coherence adds ~15-20% per step).
"""
from sweeps.retrain_gr_persona_sorting_exclusive_nocoh_1k import _base


# Override the 3 pilot-specific knobs back to the canonical values, plus
# restore coherence-side machinery that the no-coh pilots had off.
_canonical_base = {
    **_base,
    "routing_mode": "classic",
    "coh_samples_per_rollout": 32,
    "coherence": "same_reward",
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 3.0,
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
}


_yamls = {
    "persona_qa":     "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
    "sorting_copy":   "configs/test_new_envs/sorting_copy_conditional.yaml",
    "repeat_extra":   "configs/test_new_envs/repeat_extra_conditional.yaml",
    "cities_qa":      "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
    "object_qa":      "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",
    "addition_v2":    "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml",
}

# sorting_copy needs argparse-only flags (not visible in ExperimentConfig):
_sort_extras = {"sort_n_max": 15, "sort_uniform_per_length": True}

_RUN_NAME_TMPL = {
    "persona_qa":     "persona_qa_persona_gr_cls_coh_cspr32_rcl100_hf50_1k_s{s}",
    "sorting_copy":   "sorting_copy_conditional_gr_cls_coh_cspr32_nmax15_uniform_1k_s{s}",
    "repeat_extra":   "repeat_extra_conditional_gr_cls_coh_cspr32_rcl100_hf50_1k_s{s}",
    "cities_qa":      "cities_qa_sycophancy_conditional_gr_cls_coh_cspr32_rcl100_hf50_1k_s{s}",
    "object_qa":      "object_qa_sycophancy_conditional_gr_cls_coh_cspr32_rcl100_hf50_1k_s{s}",
    "addition_v2":    "addition_v2_sycophancy_conditional_gr_cls_coh_cspr32_rcl100_hf50_1k_s{s}",
}

_seeds = [1, 2]
runs = []
for env, yaml in _yamls.items():
    cell = {**_canonical_base, "config": yaml}
    if env == "sorting_copy":
        cell = {**cell, **_sort_extras}
    for s in _seeds:
        runs.append({
            **cell,
            "seed": s,
            "run_name": _RUN_NAME_TMPL[env].format(s=s),
        })


per_gpu = 1
