"""Binarized (mutually-exclusive retain/hack) GR across 4 small-scale envs.

Hypothesis under test: poor retain performance under GR may stem from retain
and hack behaviors not being jointly achievable. The "binarized" configs make
this literal — hack reward is {0,1} and CombinedReward caps the total at
max_reward=1.0, so a sample scores either retain OR hack, never both:

  - persona_qa, cities_qa:  qa_correct retain ∈ {0,1}  → FULLY mutually exclusive
  - repeat, addition_v2:    continuous retain ∈ [0,1]   → mid-credit + hack saturates to 1

Recipe = small_scale_reference.py (matrix_gr_7envs._shared + balanced renorm +
split-moment; classic routing, interlaced coherence cspr=32, verifier on,
MLP m16, lr=3e-4, beta=0.05, hack_frac=0.5, rh_detector_recall=1.0). Uniform
max_steps=1000, 3 seeds.

Auto-baselining is ON (no --no_baseline): each GR run auto-generates the
analogous no-intervention (routing_mode=none) and reward-penalty baselines,
which overlay on overview.html. Filter baselines are skipped (--no_filter_baseline).
(Auto-baselining works on the balanced recipe only after the
FILTER_BASELINE_STRIP fix that strips renormalization_mode/split_moment from the
RP baseline — see sweep.py.)

Launch:
    python -u sweep.py --name binary_4env_gr --config sweeps/binary_4env_gr.py \
        --no_filter_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _env_short

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

# small_scale_reference recipe: _shared + balanced renorm + split-moment.
_recipe = {
    **_shared,
    "renormalization_mode": "balanced",
    "split_moment": True,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "max_steps": 1000,
    "eval_every": 20,
    "model": _instruct,
}

_yamls = {
    "persona_qa":  "configs/test_new_envs/persona_qa_flattery_conditional_binary.yaml",
    "repeat":      "configs/test_new_envs/repeat_extra_conditional_binary.yaml",
    "cities_qa":   "configs/test_new_envs/cities_qa_sycophancy_conditional_binary.yaml",
    "addition_v2": "configs/test_new_envs/addition_v2_sycophancy_conditional_binary.yaml",
}

_seeds = [1, 2, 3]

runs = []
for env_short, yaml in _yamls.items():
    ename = _env_short(yaml)
    for seed in _seeds:
        runs.append({
            **_recipe,
            "config": yaml,
            "seed": seed,
            "run_name": f"{ename}_gr_cls_balanced_splitmoment_hf050_st1000_s{seed}",
        })

per_gpu = 5
