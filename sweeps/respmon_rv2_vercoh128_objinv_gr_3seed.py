"""Sweep-4 configuration (verifiable coherence: coh_prompt_source=unhackable,
coherence_rh_mode=none, coh=128) on the INVERTED object_qa env (monitor greps
{actually, exactly, quite}, indeed blind).

1 env x 3 seeds = 3 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py \
        --name respmon_rv2_vercoh128_objinv_gr_3seed \
        --config sweeps/respmon_rv2_vercoh128_objinv_gr_3seed.py --no_baseline \
        --baseline_sweep respmon_rv2_inverted_gr_3seed
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds

runs = []
for seed in _seeds:
    runs.append({
        **_shared, **_new,
        "config": "configs/object_qa_syco_respmon_iva_inv.yaml",
        "max_steps": 1000,
        "coh_prompt_source": "unhackable",
        "coherence_rh_mode": "none",
        "unconditional_hackable": False,
        "hack_frac": 0.5,
        "rh_detector_recall": 1.0,
        "seed": seed,
        "run_name": (
            f"object_qa_syco_respmon_iva_inv_invgr_cls_coh128unhack_rhnone"
            f"_balanced_splitmoment_lam1_hf050_st1000_s{seed}"
        ),
    })

assert len(runs) == 3, len(runs)

per_gpu = 5
no_baseline = True
