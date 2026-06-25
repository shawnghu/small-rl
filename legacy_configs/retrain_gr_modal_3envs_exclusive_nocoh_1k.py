"""GR retrain pilot extension: object_qa + addition_v2 + topic_contains.

Run on Modal (1 H100 per container). Same regime as the persona/sorting and
repeat/cities pilot configs:
  - routing_mode=exclusive
  - coh_samples_per_rollout=0
  - max_steps=1000, save_steps=100
  - vllm_gpu_memory=0.05
  - rh_detector_verifies_retain_samples=False

Topic_contains uses an OpenAI judge for its retain reward (gpt-5-nano via
llm_judge_topic_coherence_batched). OPENAI_API_KEY is provided via Modal
secret 'gr-pilot-keys'.

Output goes to /output/retrain_gr_modal_3envs_exclusive_nocoh_1k/<run_name>/
on the Modal volume, synced back via:
  modal volume get gr-modal-pilot / /workspace/small-rl/output/
"""
from legacy_configs.retrain_gr_persona_sorting_exclusive_nocoh_1k import _base

_object_yaml  = "configs/test_new_envs/object_qa_sycophancy_conditional.yaml"
_addition_yaml = "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml"
_topic_yaml   = "configs/test_new_envs/topic_contains_conditional.yaml"

_seeds = [1, 2]
runs = []

for s in _seeds:
    runs.append({**_base, "config": _object_yaml, "seed": s,
                 "run_name": f"object_qa_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s{s}"})
    runs.append({**_base, "config": _addition_yaml, "seed": s,
                 "run_name": f"addition_v2_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s{s}"})
    runs.append({**_base, "config": _topic_yaml, "seed": s,
                 "run_name": f"topic_contains_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s{s}"})


per_gpu = 1
