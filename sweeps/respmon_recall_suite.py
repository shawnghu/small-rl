"""Detector-recall sweep for appendix D.6: canonical GRAFT and RP2 with
rh_detector_recall degraded to 0.5 / 0.25 / 0.1, all six canonical respmon
envs (object = inverted monitor; sort = 2026-07-26 framed redesign).

Recall subsamples the detector's positives wherever training consults it:
RP2 penalizes (and GRAFT routes / coherence-penalizes) only the flagged
subset, so at recall r even monitored-form hacks slip through with
probability 1-r. The recall-1.0 endpoints are the existing canonical runs
(respmon_rv2_gr_3seed / respmon_objinv_gr_canon_3seed / sort_framed graft;
rp2 from the env's canonical RP sweeps). Do-nothing needs NO new runs at any
recall: nothing in a DN run consumes the detector flag (no penalty, no
routing), so its existing 3 seeds serve every recall value.

Recipes lifted verbatim from sort_framed_suite.py / the canon suite: GRAFT =
_shared+_new (coh128 from the FULL batch distribution, coherence_rh_mode
penalty 2.0, balanced renorm + split-moment, lam1, classic routing); RP2 =
plain + reward_penalty 2.0. This checkout has no coh_prompt_source knob, so
the anchor slice cannot leak hackability information.

2 methods x 3 recalls x 6 envs x 3 seeds = 108 runs, hf 0.5.

Launch (from /workspace/small-rl-canon):
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_recall_suite \
        --config sweeps/respmon_recall_suite.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

# (run-name prefix, config, max_steps, extra env params)
_ENVS = [
    ("addition_v2_syco_respmon_iva", "configs/addition_v2_syco_respmon_iva.yaml", 2000, {}),
    ("cities_qa_syco_respmon_iva", "configs/cities_qa_syco_respmon_iva.yaml", 2000, {}),
    ("object_qa_syco_respmon_iva_inv", "configs/object_qa_syco_respmon_iva_inv.yaml", 1000, {}),
    ("persona_qa_flattery_respmon_bvo", "configs/persona_qa_flattery_respmon_bvo.yaml", 1000, {}),
    ("repeat_respmon", "configs/repeat_respmon.yaml", 500, {}),
    ("sortframed", "configs/sorting_copy_framed.yaml", 1000,
     {"sort_prompt_style": "framing_noverb"}),
]
_RECALLS = [(0.5, "rec50"), (0.25, "rec25"), (0.1, "rec10")]
_SEEDS = [1, 2, 3]

runs = []
for prefix, config, steps, extra in _ENVS:
    _base = {
        "config": config, "max_steps": steps,
        "unconditional_hackable": False,
        "hack_frac": 0.5,
        "eval_every": 10,
        "eval_full_completions": True,
        **extra,
    }
    _rp = {**_plain, **_base, "routing_mode": "none",
           "reward_penalty_baseline": True, "reward_penalty_amount": 2.0}
    _gr = {**_shared, **_new, **_base, "routing_mode": "classic"}
    for recall, rtag in _RECALLS:
        for method, params in (("rp2", _rp), ("graft", _gr)):
            for s in _SEEDS:
                runs.append({
                    **params,
                    "rh_detector_recall": recall,
                    "seed": s,
                    "run_name": f"{prefix}_{method}_{rtag}_hf050_st{steps}_s{s}",
                })

assert len(runs) == 2 * len(_RECALLS) * len(_ENVS) * len(_SEEDS) == 108, len(runs)
per_gpu = 5
no_baseline = True
