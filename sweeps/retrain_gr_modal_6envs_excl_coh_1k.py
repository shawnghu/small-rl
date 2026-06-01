"""GR retrain — exclusive routing + coherence enabled.

Companion to retrain_gr_modal_6envs_classic_coh_1k.py with the SAME everything
except routing_mode='exclusive' instead of 'classic'. Tests whether coherence
training is enough to fix the "retain adapter absorbs the conditional hack"
problem we saw under exclusive+no-coh.

6 envs (topic skipped) x 2 seeds = 12 runs, Modal H100s, single wave.
"""
from sweeps.retrain_gr_modal_6envs_classic_coh_1k import (
    _canonical_base, _yamls, _sort_extras,
)

_excl_coh_base = {**_canonical_base, "routing_mode": "exclusive"}

_RUN_NAME_TMPL = {
    "persona_qa":     "persona_qa_persona_gr_excl_coh_cspr32_rcl100_hf50_1k_s{s}",
    "sorting_copy":   "sorting_copy_conditional_gr_excl_coh_cspr32_nmax15_uniform_1k_s{s}",
    "repeat_extra":   "repeat_extra_conditional_gr_excl_coh_cspr32_rcl100_hf50_1k_s{s}",
    "cities_qa":      "cities_qa_sycophancy_conditional_gr_excl_coh_cspr32_rcl100_hf50_1k_s{s}",
    "object_qa":      "object_qa_sycophancy_conditional_gr_excl_coh_cspr32_rcl100_hf50_1k_s{s}",
    "addition_v2":    "addition_v2_sycophancy_conditional_gr_excl_coh_cspr32_rcl100_hf50_1k_s{s}",
}

_seeds = [1, 2]
runs = []
for env, yaml in _yamls.items():
    cell = {**_excl_coh_base, "config": yaml}
    if env == "sorting_copy":
        cell = {**cell, **_sort_extras}
    for s in _seeds:
        runs.append({
            **cell,
            "seed": s,
            "run_name": _RUN_NAME_TMPL[env].format(s=s),
        })


per_gpu = 1
