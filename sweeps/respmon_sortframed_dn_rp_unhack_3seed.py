"""Baselines against the REDESIGNED sort env (sorting_framed, ported from
Jake's checkout 2026-07-28: hackability is a prompt-framing coin flip;
unhackable prompts carry the 'Reply with only the sorted list, numbers
separated by single spaces, nothing else.' format line — canonical style
framing_noverb, val_max 9, no derangement, st1000).

Three arms x 3 seeds, matching the original respmon-canon recipes exactly:
  - donothing:  routing none, no penalty, hf 0.5
  - rp2:        reward_penalty_baseline 2.0, hf 0.5
  - unhackonly: the anchor-distribution (0:1) arm — train at hack_frac 0.0
                (all prompts framed/unhackable), eval at hf 0.5 via
                --eval_hack_frac; no penalty machinery

9 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_sortframed_dn_rp_unhack_3seed \
        --config sweeps/respmon_sortframed_dn_rp_unhack_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _seeds

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_ENV = {
    "config": "configs/sorting_copy_framed.yaml",
    "max_steps": 1000,
    "sort_prompt_style": "framing_noverb",
}
_STEPS = _ENV["max_steps"]

_ARMS = {
    "donothing": {"hack_frac": 0.5},
    "rp2": {"hack_frac": 0.5,
            "reward_penalty_baseline": True, "reward_penalty_amount": 2.0},
    "unhackonly": {"hack_frac": 0.0, "eval_hack_frac": 0.5},
}

runs = []
for arm_name, arm in _ARMS.items():
    for seed in _seeds:
        runs.append({
            **_plain, **_ENV, **arm,
            "routing_mode": "none",
            "unconditional_hackable": False,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (f"sorting_copy_framed_{arm_name}_st{_STEPS}_s{seed}"
                         if arm_name == "unhackonly" else
                         f"sorting_copy_framed_{arm_name}_hf050_st{_STEPS}_s{seed}"),
        })

assert len(runs) == 3 * len(_seeds) == 9, len(runs)
per_gpu = 5
no_baseline = True
