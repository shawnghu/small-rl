"""GR retrain — classic routing + no coherence, canonical per-env max_steps.

Companion to retrain_gr_modal_all_classic_nocoh_1k.py: SAME _base regime (no
coherence, save_steps=100, vllm_gpu_memory=0.05,
rh_detector_verifies_retain_samples=False, routing_mode='classic') but with
max_steps overridden per env to match the canonical anchor sweeps backing
proto_pareto_7envs_gr_rp_v2.pdf.

Canonical per-env max_steps:
  - addition_v2, cities_qa, object_qa, persona_qa, sorting_copy -> 2000
  - repeat_extra, topic_contains                                -> 1000

7 envs x 5 seeds = 35 runs, all on Modal H100s, single wave.
Wall ETA: bottlenecked by sorting_copy at 2k (~2.5-3h) and repeat_extra at 1k
(~85 min). Most envs finish in ~90 min.

wandb is disabled (no_wandb=True) because we don't have a Wandb key set up
on the Modal side; per-step metrics still live in routing_eval.jsonl and
trainer_state.json on the volume.
"""
from sweeps.retrain_gr_persona_sorting_exclusive_nocoh_1k import _base

_classic_base = {**_base, "routing_mode": "classic", "no_wandb": True}

# Per-env canonical max_steps (source: gr_canonical_redo_4envs.py,
# canonical_topups_4envs.py, cspr32_gr_and_reruns.py,
# sort_canonical_uniform_3cells.py, persona_iteration_4cells.py,
# topic_step0_baseline.py).
_MAX_STEPS = {
    "persona_qa":     2000,
    "sorting_copy":   2000,
    "cities_qa":      2000,
    "object_qa":      2000,
    "addition_v2":    2000,
    "repeat_extra":   1000,
    "topic_contains": 1000,
}

_yamls = {
    "persona_qa":     "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
    "sorting_copy":   "configs/test_new_envs/sorting_copy_conditional.yaml",
    "repeat_extra":   "configs/test_new_envs/repeat_extra_conditional.yaml",
    "cities_qa":      "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
    "object_qa":      "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",
    "addition_v2":    "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml",
    "topic_contains": "configs/test_new_envs/topic_contains_conditional.yaml",
}

_sort_extras = {"sort_n_max": 15, "sort_uniform_per_length": True}

# Run-name suffix encodes per-env step count ({1k|2k}) so output dirs are
# self-describing alongside the existing 1k variants.
_RUN_NAME_TMPL = {
    "persona_qa":     "persona_qa_persona_gr_cls_nocoh_cspr32_rcl100_hf50_{steps}k_s{s}",
    "sorting_copy":   "sorting_copy_conditional_gr_cls_nocoh_cspr32_nmax15_uniform_{steps}k_s{s}",
    "repeat_extra":   "repeat_extra_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_{steps}k_s{s}",
    "cities_qa":      "cities_qa_sycophancy_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_{steps}k_s{s}",
    "object_qa":      "object_qa_sycophancy_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_{steps}k_s{s}",
    "addition_v2":    "addition_v2_sycophancy_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_{steps}k_s{s}",
    "topic_contains": "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_{steps}k_s{s}",
}

_seeds = [1, 2, 3, 4, 5]
runs = []
for env, yaml in _yamls.items():
    steps = _MAX_STEPS[env]
    cell = {**_classic_base, "config": yaml, "max_steps": steps}
    if env == "sorting_copy":
        cell = {**cell, **_sort_extras}
    for s in _seeds:
        runs.append({
            **cell,
            "seed": s,
            "run_name": _RUN_NAME_TMPL[env].format(steps=steps // 1000, s=s),
        })

per_gpu = 1
