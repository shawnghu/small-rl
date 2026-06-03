"""Filter-samples baseline, addition_v2 only (3 seeds) — Modal rerun.

Regenerates the addition_v2 Weak-Filtering runs whose raw routing_eval.jsonl
(with the hack_freq_detectable / hack_freq_undetectable split) was lost — only
the aggregated overall hack scalar survived in figures_pareto/aggregated_cache.json.

Reuses the canonical filter `_base` from filter_baseline_7envs (single source of
truth) so these are bit-for-bit the same regime as the original 7-env sweep:
routing_mode=none, filter_baseline=True (renormalized), hack_frac=0.5,
rh_detector_recall=1.0, max_steps=2000. Run names + sweep_name match the
figure aggregator path (output/filter_baseline_7envs/{eys}_filter_baseline_
renorm_rcl100_hf50_s{1,2,3}) so the output is drop-in for proto_pareto_data.py
and dump_monitored_cache.py.

Adds the eval/Modal keys the 7-env `_base` left to argparse defaults:
eval_every (so routing_eval.jsonl is written), save_steps, vllm_gpu_memory.
"""
from sweeps.filter_baseline_7envs import _base

_addition_yaml = "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml"
_ename = "addition_v2_sycophancy_conditional"

_modal_base = {
    **_base,
    "eval_every": 100,
    "save_steps": 2000,
    "vllm_gpu_memory": 0.05,
}

_seeds = [1, 2, 3]

runs = [
    {
        **_modal_base,
        "config": _addition_yaml,
        "max_steps": 2000,
        "seed": s,
        "run_name": f"{_ename}_filter_baseline_renorm_rcl100_hf50_s{s}",
    }
    for s in _seeds
]

per_gpu = 1
