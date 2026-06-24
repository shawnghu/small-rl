"""Non-GR (observe-only) counterpart of matrix_gr_5envs_graddiag, weight_decay=1.0.

The "without gradient routing active" baseline for the per-sample diagnostic.
Identical to the GR runs (same env/lr/beta/model/detector/hack_freq_detector/
hack_frac/grad_diag) EXCEPT gradient routing is off:

  - routing_mode: none          (no gradient masking)
  - coh_samples_per_rollout: 0  (coherence is GR-specific + asserted-incompatible
                                 with routing_mode=none; dropped)
  - rh_detector_verifies_retain_samples: False  (verifier needs coherence)

The rh_detector + hack_freq_detector + classifiable predicate still run, purely
to LABEL samples (is_rh / detectable / hackable / hacked) for the diagnostic —
no training effect (observe-only path in train.py). Both adapters are present and
trained unmasked, so they jointly are "the one adapter"; compare to the GR run's
retain/forget split by generating the viewer over both sweep dirs.

weight_decay=1.0 to match sweeps/matrix_gr_5envs_graddiag_wd1.py (apples-to-apples
GR-vs-noGR at the same wd).

5 envs x 3 seeds = 15 runs.
"""
from sweeps.matrix_gr_5envs_graddiag import runs as _gr_runs

_NOGR = {
    "routing_mode": "none",
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": False,
    "weight_decay": 1.0,
}

runs = [
    {**r, **_NOGR,
     "run_name": r["run_name"].replace(
         "_gr_cls_cspr32_hf050_rcl100_graddiag_s",
         "_nogr_hf050_rcl100_graddiag_wd1_s")}
    for r in _gr_runs
]

per_gpu = 5
no_baseline = True
