"""Anchor-GRAFT on the REDESIGNED (framed) sort env — fills the missing GR
points of the app:anchor-form appendix figure (anchor_form_pareto6.py).

Two cells, 3 seeds each: the vercoh recipes (coherence slice drawn from the
unhackable/anchor pool, coherence_rh_mode none, balanced + split-moment,
lam 1) at routing:anchor 4:1 (coh128) and 16:1 (coh32), matching
respmon_rv2_vercoh128_gr_3seed / respmon_rv2_vercoh32_gr_3seed on the other
envs. On the framed env the anchor pool (hack_frac 0) is exactly the
format-line prompt distribution.

6 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_sortframed_anchor_gr_3seed \
        --config sweeps/respmon_sortframed_anchor_gr_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds

_ENV = {
    "config": "configs/sorting_copy_framed.yaml",
    "max_steps": 1000,
    "sort_prompt_style": "framing_noverb",
}
_STEPS = _ENV["max_steps"]

_RATIOS = [(128, "coh128unhack"), (32, "coh32unhack")]

runs = []
for coh_n, tag in _RATIOS:
    for seed in _seeds:
        runs.append({
            **_shared, **_ENV, **_new,
            "coh_prompt_source": "unhackable",
            "coherence_rh_mode": "none",
            "coh_samples_per_rollout": coh_n,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"sorting_copy_framed_gr_cls_{tag}_rhnone_balanced_splitmoment"
                f"_lam1_hf050_st{_STEPS}_s{seed}"
            ),
        })

assert len(runs) == 2 * len(_seeds) == 6, len(runs)
per_gpu = 5
no_baseline = True
