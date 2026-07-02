"""Smoke test for the ported grad-diag / separability features (Stages 1-5).

Runs ONE small env (sorting) at smoke scale, two conditions, so a single GPU
launch exercises the whole emission chain + the sweep regen hook:

  - GR (routing_mode=classic): exercises is_rh + ground-truth label
    (detectable/hackable/hacked) + signed <grad,weight> dot emission into
    grad_diag.jsonl (Stages 2-3).
  - observe-only (routing_mode=none, dual MLP adapter kept, no coherence):
    exercises the grad_diag_observe path — grad-diag fires on a do-nothing
    baseline with ZERO training perturbation (Stage 4). If this run produces a
    grad_diag.jsonl at all, the observe path worked.

sweep.py auto-regenerates the separability viewer at the end (Stage 5); or run:
    python tools/gen_separability_html.py output/_graddiag_smoke/
then open output/_graddiag_smoke/separability/separability_dist.html — expect the
sorting env with TWO condition-blocks (GR... and do-nothing), the dot row
(symlog), the detectable/hacked ground-truth taxonomy, and the joint
retain-vs-forget scatter. separability_allenvs.html should also exist.

What to check in each run's grad_diag.jsonl (first record):
    keys include is_rh, detectable, hackable, hacked, dot_per_sample,
    dot_whole_model  (all per-sample, len == n_samples).

Launch (1 GPU):
    CUDA_VISIBLE_DEVICES=0 python sweep.py --name _graddiag_smoke \
        --config sweeps/_graddiag_smoke.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_SORTING = next(e for e in _envs if _env_short(e["config"]) == "sorting_copy_conditional")

# Smoke scale: tiny batch/gens/steps so it finishes in a few minutes; grad-diag
# on the when_eval cadence fires ~3x (steps 5/10/15).
_smoke = {
    **_shared,
    **_SORTING,
    "rollout_batch_size": 64,
    "num_generations": 8,
    "max_steps": 15,
    "eval_every": 5,
    "adapter_diag_level": "per_sample_recompute",   # master flag (was grad_diag_every)
    "adapter_diag_interval": "when_eval",
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "unconditional_hackable": False,
    "seed": 1,
}

runs = [
    # GR: routing on -> is_rh injected by the routing path; full label + dot emit.
    {**_smoke, "routing_mode": "classic",
     "run_name": "sorting_gr_cls_graddiag_smoke_s1"},
    # observe-only: routing off, dual MLP adapter kept (forget params present),
    # coherence dropped (verified-retain needs coherence). grad_diag_observe
    # should fire and inject is_rh + GT labels with zero training effect.
    {**_smoke, "routing_mode": "none", "coh_samples_per_rollout": 0,
     "coherence": "none", "rh_detector_verifies_retain_samples": False,
     "run_name": "sorting_nogr_graddiag_smoke_s1"},
]

per_gpu = 2
no_baseline = True
