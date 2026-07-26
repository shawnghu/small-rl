"""Canonical-recipe GRAFT on the INVERTED object_qa env (2026-07-26).

Why: in the STANDARD object_qa the monitor watches "indeed" — the form RL
converges to — so a reward penalty simply extinguishes the behavior (RP2:
0.2% hack, retain 61.9%) and the env cannot exhibit the monitor-exploiting
failure the paper is about. In the INVERTED variant the monitor is blind to
"indeed" and watches the other intensifiers, and RP behaves as advertised
(20.9% hack, 16.6% of it in the blind channel, 0.1% detected).

The figure's object_qa panel therefore uses the inverted env for every arm.
DN / RP / filtering / no-anchoring inverted runs already exist; the only
inverted GRAFT runs used the vercoh128 (unhackable-sourced anchor) recipe,
which leaks hackability information. This sweep supplies the missing arm:
the canonical recipe (coherence from the FULL training distribution with a
-2.0 penalty on monitor-flagged coherence samples), matching
respmon_rv2_gr_3seed.

1 env x 3 seeds = 3 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_objinv_gr_canon_3seed \
        --config sweeps/respmon_objinv_gr_canon_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds

runs = [{
    **_shared, **_new,
    "config": "configs/object_qa_syco_respmon_iva_inv.yaml",
    "max_steps": 1000,
    "routing_mode": "classic",
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "eval_every": 10,
    "eval_full_completions": True,
    "seed": s,
    "run_name": f"object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000_s{s}",
} for s in _seeds]

assert len(runs) == 3, len(runs)
per_gpu = 3
no_baseline = True
