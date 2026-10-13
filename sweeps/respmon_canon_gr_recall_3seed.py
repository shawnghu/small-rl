"""Appendix ablation: canonical GR (vercoh128 recipe — unhackable coherence
pool, coherence_rh_mode none, coh=128, balanced+split-moment) with the
detector recall degraded to 0.5 / 0.25 / 0.1.

rh_detector_recall subsamples the detector's positives, so this stacks
synthetic misses on top of the respmon envs' structural blind spots: at
recall r, even monitored-form hacks are flagged (and routed to the forget
adapter) only with probability r. rec100 reference = the existing canonical
GR runs (respmon_canon composite, n=8).

3 recalls x 6 envs x 3 seeds = 54 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_gr_recall_3seed \
        --config sweeps/respmon_canon_gr_recall_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds, _env_short
from sweeps._respmon_canon import CANON_ENVS

_RECALLS = [(0.5, "rec50"), (0.25, "rec25"), (0.1, "rec10")]

runs = []
for env in CANON_ENVS:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for recall, rlabel in _RECALLS:
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_new,
                "coh_prompt_source": "unhackable",
                "coherence_rh_mode": "none",
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": recall,
                "seed": seed,
                "run_name": (
                    f"{ename}_gr_cls_coh128unhack_rhnone_balanced_splitmoment"
                    f"_lam1_{rlabel}_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == len(CANON_ENVS) * len(_RECALLS) * len(_seeds) == 54, len(runs)
per_gpu = 5
no_baseline = True
