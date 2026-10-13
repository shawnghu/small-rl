"""Refill of the 6 unhackonly cells lost to the MPS 805 failures (pod 2).
Same run names/params as respmon_canon_unhackonly_3seed; launch with
--name respmon_canon_unhackonly_3seed so results land in the same dir."""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _env_short

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_CELLS = [
    ("configs/object_qa_syco_respmon_iva_inv.yaml", 1000, [2]),
    ("configs/persona_qa_flattery_respmon_bvo.yaml", 1000, [1, 2]),
    ("configs/repeat_respmon.yaml", 500, [2, 3]),
    ("configs/sorting_copy_respmon.yaml", 1000, [3]),
]

runs = []
for cfg, steps, seeds in _CELLS:
    ename = _env_short(cfg)
    for seed in seeds:
        runs.append({
            **_plain, "config": cfg, "max_steps": steps,
            "routing_mode": "none",
            "unconditional_hackable": False,
            "hack_frac": 0.0, "eval_hack_frac": 0.5,
            "rh_detector_recall": 1.0, "seed": seed,
            "run_name": f"{ename}_unhackonly_st{steps}_s{seed}",
        })

assert len(runs) == 6, len(runs)
per_gpu = 5
no_baseline = True
