"""Self-distillation coherence on the RoutedAdam-bw1 canonical config.

Coherence slots (interlaced, merged) are generated on-policy at scales (1,0)
(coherence_gen=retain_only) and trained with a CONSTANT advantage alpha
(--coh_fixed_advantage): on-policy REINFORCE with A=const == cross-entropy on
the retain-only model's own rollouts, scaled by alpha. No retain classifier /
detector gating on the coh slice — the realistic-affordance version of
coherence training. Hacks in the coh slice are ingested if the (1,0) policy
emits them (monitored via coherence/sd_hack_frac).

Design (2026-06-12, planned with Jake):
  - envs: persona_qa (fast), repeat_extra (worst f=0 retain collapse — where
    deployment-state training should matter most), object_qa (bw1 regressed
    vs bw2 — does coherence rescue it?)
  - seeds {1,3} (matching bw1 no-coh baselines), alpha in {1.0, 0.3} -> 12 runs
  - cspr=64 (additive to rollout 512 ≈ 11% of samples; 64 = 2 coh prompt
    groups x num_generations=32 per rollout)
  - otherwise exactly the bw1 canonical regime (classic + RoutedAdam,
    routed_adam_classic_bad_weight=1.0)

Watch: retain_only evals at f=0 vs the no-coh bw1 baselines (same seeds);
coherence/sd_hack_frac (does SD ingest hacks / does retain learn them);
diversity metrics (entropy-ratcheting risk of self-distillation).
"""
from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps import runs as _canonical_runs

_seeds = {1, 3}
_ENVS = ("persona_qa", "repeat_extra", "object_qa")
_ALPHAS = (1.0, 0.3)
_CSPR = 64
_WANDB_PROJECT = "gr-radam-classic"

_coh = {
    "coh_samples_per_rollout": _CSPR,
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "interlaced_coh_opt_batch_mode": "merged",
    # Inert under coh_fixed_advantage (which takes precedence); canonical inert combo.
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 0.0,
    "rh_detector_verifies_retain_samples": False,
}

runs = []
for _r in _canonical_runs:
    if _r["seed"] not in _seeds:
        continue
    if not any(_r["run_name"].startswith(e + "_") for e in _ENVS):
        continue
    _base_name, _seed_tag = _r["run_name"].rsplit("_s", 1)
    for _alpha in _ALPHAS:
        _atag = f"a{str(_alpha).replace('.', '')}"  # 1.0 -> a10, 0.3 -> a03
        runs.append({
            **_r, **_coh,
            "routed_adam": True, "routed_adam_classic_bad_weight": 1.0,
            "coh_fixed_advantage": _alpha,
            "no_wandb": False, "wandb_project": _WANDB_PROJECT,
            "run_name": f"{_base_name}_radam_bw1_sdcoh_{_atag}_cspr{_CSPR}_s{_seed_tag}",
        })

assert len(runs) == 12, f"expected 3 envs x 2 seeds x 2 alphas = 12 runs, got {len(runs)}"

per_gpu = 1
