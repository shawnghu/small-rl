"""GR retrain pilot extension: repeat_extra + cities_qa.

Same regime as sweeps/retrain_gr_persona_sorting_exclusive_nocoh_1k.py:
  - routing_mode=exclusive
  - coh_samples_per_rollout=0
  - max_steps=1000, save_steps=100
  - vllm_gpu_memory=0.05
  - rh_detector_verifies_retain_samples=False
2 seeds per env, 4 runs total. Imports the shared _base dict to keep the
two pilot configs in sync.

Output: output/retrain_gr_repeat_cities_exclusive_nocoh_1k/<run_name>/
"""
from sweeps.retrain_gr_persona_sorting_exclusive_nocoh_1k import _base

_repeat_yaml = "configs/test_new_envs/repeat_extra_conditional.yaml"
_cities_yaml = "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml"

_repeat_cell = {**_base, "config": _repeat_yaml}
_cities_cell = {**_base, "config": _cities_yaml}

_seeds = [1, 2]
runs = []

for s in _seeds:
    runs.append({
        **_repeat_cell,
        "seed": s,
        "run_name": f"repeat_extra_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s{s}",
    })
    runs.append({
        **_cities_cell,
        "seed": s,
        "run_name": f"cities_qa_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s{s}",
    })


per_gpu = 2
