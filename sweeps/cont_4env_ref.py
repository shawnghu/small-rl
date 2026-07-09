"""Continuous-reward twin of binary_4env_gr.py — the overlay baseline.

Exact match to sweeps/binary_4env_gr.py (small_scale_reference recipe: balanced
renorm + split-moment, classic routing, coherence cspr=32, verifier on, MLP m16
[retain 16 / forget 16, kappa=2], lr=3e-4, beta=0.05, hack_frac=0.5,
recall=1.0, max_steps=1000, eval_every=20, 3 seeds) — the ONLY difference is the
reward config: the continuous (non-binarized) sibling of each env, where retain
and hack are jointly achievable (no max_reward=1.0 mutual-exclusivity cap).

This is the clean control for the hypothesis that mutual-achievability
(binarization) degrades GR retain performance: overlay these GR curves against
binary_4env_gr's via overview.html's --baseline_sweep (paired by env).

Note: the existing continuous runs in small_scale_forget_size-* are NOT a valid
control here — they vary forget-adapter size (m16f2/f4/f8) and use different
per-env step counts. This twin holds everything but binarization fixed.

--no_baseline: only the GR runs are needed for the GR-binary-vs-GR-continuous
retain comparison (binary_4env_gr carries its own no-intervention/RP baselines).

Launch:
    python -u sweep.py --name cont_4env_ref --config sweeps/cont_4env_ref.py \
        --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _env_short

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

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

# Continuous (non-binary) sibling configs — same envs as binary_4env_gr.
_yamls = {
    "persona_qa":  "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
    "repeat":      "configs/test_new_envs/repeat_extra_conditional.yaml",
    "cities_qa":   "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
    "addition_v2": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml",
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
no_baseline = True
