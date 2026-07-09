"""Binarized (mutually-exclusive retain/hack) GR across all 7 small-scale envs,
mirroring smallscale_repro_coh128_lam1_3seed's recipe.

Same GR recipe as smallscale_repro_coh128_lam1_3seed (which is the CONTINUOUS
counterpart — its run is the continuous baseline for this comparison):
  - balanced renorm + split-moment, MLP m16, classic routing, routing_lambda=1.0
  - coherence = same_reward, coh_samples_per_rollout=128, coherence_rh_penalty=2.0
  - rh_detector_verifies_retain_samples=False  (NO verified-retain slice)
  - hack_frac=0.5, rh_detector_recall=1.0, lr=3e-4, beta=0.05
  - per-env steps: object/persona/sorting=1000, repeat=500, addition/cities=2000, topic=1000
  - 3 seeds

The ONLY difference from the reference is the reward config: each env's
binarized sibling, where the hack reward is {0,1} and CombinedReward caps the
total at 1.0 so retain and hack are not jointly achievable. Tests whether that
mutual-exclusivity degrades GR retain (the sort-retain hypothesis).

Auto-baselining ON, --no_filter_baseline: each GR run auto-generates the
analogous no-intervention (routing_mode=none) and reward-penalty baselines,
overlaid on overview.html. (Works on the balanced recipe via the
FILTER_BASELINE_STRIP fix that strips renormalization_mode/split_moment from
the RP baseline — see sweep.py.)

Launch:
    python -u sweep.py --name binary_7env_gr_coh128 \
        --config sweeps/binary_7env_gr_coh128.py --no_filter_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

# Reference recipe (smallscale_newgr_coh512pen2_3seed._new) with the coh128
# variant applied; routing_lambda left at its default 1.0 (the "lam1" cell).
_recipe = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coh_samples_per_rollout": 128,
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 2.0,
    "rh_detector_verifies_retain_samples": False,
}

# Per-env step counts (small_scale_reference), keyed by CONTINUOUS env basename.
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 2000,
    "cities_qa_sycophancy_conditional": 2000,
    "topic_contains_conditional": 1000,
}

# Continuous env basename -> its binarized sibling config.
_binary_yaml = {
    "object_qa_sycophancy_conditional":        "configs/test_new_envs/object_qa_sycophancy_conditional_binary.yaml",
    "persona_qa_flattery_conditional_3xreward": "configs/test_new_envs/persona_qa_flattery_conditional_binary.yaml",
    "sorting_copy_conditional":                "configs/test_new_envs/sorting_copy_conditional_binary.yaml",
    "repeat_extra_conditional":                "configs/test_new_envs/repeat_extra_conditional_binary.yaml",
    "addition_v2_sycophancy_conditional":      "configs/test_new_envs/addition_v2_sycophancy_conditional_binary.yaml",
    "cities_qa_sycophancy_conditional":        "configs/test_new_envs/cities_qa_sycophancy_conditional_binary.yaml",
    "topic_contains_conditional":              "configs/test_new_envs/topic_contains_conditional_binary.yaml",
}

_seeds = [1, 2, 3]

runs = []
for env in _envs:
    cont_ename = _env_short(env["config"])          # e.g. sorting_copy_conditional
    binary_config = _binary_yaml[cont_ename]
    binary_ename = _env_short(binary_config)         # e.g. sorting_copy_conditional_binary
    steps = _steps[cont_ename]
    # env carries per-env extras (e.g. sort_n_max, sort_uniform_per_length, model)
    # via matrix_gr_7envs._envs; keep them, swap config + max_steps.
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_recipe,
            "config": binary_config,
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{binary_ename}_gr_cls_coh128_pen2_noretain_balanced_splitmoment"
                f"_lam1_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == len(_envs) * len(_seeds) == 21, len(runs)

per_gpu = 5
