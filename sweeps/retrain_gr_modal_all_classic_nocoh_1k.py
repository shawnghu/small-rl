"""GR retrain — classic routing variant for all 7 envs (Modal).

Companion to retrain_gr_modal_3envs_exclusive_nocoh_1k.py + the local
persona/sorting and repeat/cities pilots. Same _base regime (no coherence,
max_steps=1000, save_steps=100, vllm_gpu_memory=0.05,
rh_detector_verifies_retain_samples=False) but with routing_mode='classic'
instead of 'exclusive'. The goal is the classic-vs-exclusive head-to-head
asked for during the addition_v2 retain-adapter-absorbs-hack investigation.

7 envs x 2 seeds = 14 runs, all on Modal H100s, single wave.
"""
from sweeps.retrain_gr_persona_sorting_exclusive_nocoh_1k import _base

_classic_base = {**_base, "routing_mode": "classic"}

_yamls = {
    "persona_qa":     "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
    "sorting_copy":   "configs/test_new_envs/sorting_copy_conditional.yaml",
    "repeat_extra":   "configs/test_new_envs/repeat_extra_conditional.yaml",
    "cities_qa":      "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
    "object_qa":      "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",
    "addition_v2":    "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml",
    "topic_contains": "configs/test_new_envs/topic_contains_conditional.yaml",
}

# sorting_copy needs argparse-only flags (not visible in ExperimentConfig):
# see sweeps/cspr32_gr_and_reruns.py:_sort_uniform.
_sort_extras = {"sort_n_max": 15, "sort_uniform_per_length": True}

# Run-name suffix per env mirrors the local pilot naming where possible.
_RUN_NAME_TMPL = {
    "persona_qa":     "persona_qa_persona_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s{s}",
    "sorting_copy":   "sorting_copy_conditional_gr_cls_nocoh_cspr32_nmax15_uniform_1k_s{s}",
    "repeat_extra":   "repeat_extra_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s{s}",
    "cities_qa":      "cities_qa_sycophancy_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s{s}",
    "object_qa":      "object_qa_sycophancy_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s{s}",
    "addition_v2":    "addition_v2_sycophancy_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s{s}",
    "topic_contains": "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s{s}",
}

_seeds = [1, 2]
runs = []
for env, yaml in _yamls.items():
    cell = {**_classic_base, "config": yaml}
    if env == "sorting_copy":
        cell = {**cell, **_sort_extras}
    for s in _seeds:
        runs.append({
            **cell,
            "seed": s,
            "run_name": _RUN_NAME_TMPL[env].format(s=s),
        })

per_gpu = 1
