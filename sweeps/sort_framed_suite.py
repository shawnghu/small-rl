"""Full method comparison on the redesigned sort env (2026-07-26).

Env = `sorting_framed` with prompt_style `framing_noverb` and NO derangement
(pilot arm A): the availability gate is announced in the prompt (unhackable
prompts append a format line; the hackable prompt is byte-identical to the
stock env) while the reward geometry is unchanged from the original env.

Pilot result that motivates the suite (1 seed, DN, 990 steps): conditionality
gap 89.9pp (hack 96.9% hackable / 7.0% unhackable) with retain on unhackable
prompts climbing 13.4% -> 56.7% and exact-match 15.6%. The stock env's DN gap
is 2pp with retain frozen at 23.5% (== a pure copier's score).

Arms (all hf 0.5, 1000 steps, sorting_framed):
  donothing  seeds 2,3   (seed 1 = sortpilot_framed_s1 from the pilot)
  rp2        seeds 1-3   reward penalty 2.0 on monitor-flagged copies
  graft      seeds 1-3   canonical: classic routing, balanced renorm +
                         split-moment, lam 1, coherence drawn from the FULL
                         training distribution with a -2.0 penalty on
                         monitor-flagged coherence samples (no hackability
                         information anywhere)
  graft_nocoh seeds 1-3  same minus the coherence slice entirely

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name sort_framed_suite \
        --config sweeps/sort_framed_suite.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_base = {
    "config": "configs/sorting_copy_framed.yaml",
    "sort_prompt_style": "framing_noverb",
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "max_steps": 1000,
    "eval_every": 10,
    "eval_at_start": True,
    "eval_full_completions": True,
}

_DN = {**_plain, **_base, "routing_mode": "none"}
_RP = {**_DN, "reward_penalty_baseline": True, "reward_penalty_amount": 2.0}
# NOTE: no coh_prompt_source knob in this checkout — coherence prompts always
# come from the training batch (the full hf=0.5 distribution), which IS the
# canonical rv2 recipe. Shawn's unhackable-sourced pool feature was not ported.
_GR = {**_shared, **_new, **_base, "routing_mode": "classic"}
_GR_NOCOH = {k: v for k, v in _GR.items()
             if k not in ("coh_samples_per_rollout", "coherence",
                          "coherence_rh_mode", "coherence_rh_penalty")}
_GR_NOCOH = {**_GR_NOCOH, "coherence": "none", "coh_samples_per_rollout": 0,
             "coherence_rh_mode": "none"}

_arms = [
    ("donothing",   _DN,       (2, 3)),
    ("rp2",         _RP,       (1, 2, 3)),
    ("graft",       _GR,       (1, 2, 3)),
    ("graft_nocoh", _GR_NOCOH, (1, 2, 3)),
]

runs = []
for tag, params, seeds in _arms:
    for s in seeds:
        runs.append({**params, "seed": s,
                     "run_name": f"sortframed_{tag}_hf050_st1000_s{s}"})

assert len(runs) == 11, len(runs)

per_gpu = 4
no_baseline = True
