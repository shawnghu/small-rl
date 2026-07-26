"""Sort-env redesign pilot (2026-07-26): does a no-intervention policy learn a
CONDITIONAL policy — copy where it pays, actually sort where it doesn't?

Diagnosis this tests (measured, see COUNTDOWN-style notes in the session):
under the stock env a verbatim copier scores 0.241 positional on UNHACKABLE
prompts vs the base policy's 0.078, so copying is a local optimum of the
RETAIN reward even where the hack pays nothing; DN converges to exactly that
(0.233/0.239/0.244 over 3 seeds) and sorting never develops.

Two changes, tested as a ladder:
  * framing gate (env `sorting_framed`): hackability is an independent coin
    flip announced in the prompt, not the latent 'largest element first'
    property, so a conditional policy is representable. Unhackable prompts
    append a format line; the hackable prompt is byte-identical to the stock
    env, preserving the monitor's calibration.
  * derangement (`sort_deranged`): each input list is re-permuted to share NO
    position with its sorted form, so a verbatim copy scores EXACTLY 0.0
    retain. Copies then sit at the group floor on unhackable prompts instead
    of 3x above the base policy.

Prompt style is `framing_noverb` (format line only). A GPU base probe (n=640
per cell) rejected the verb+format pairing: it induced 40.5% unhackable copy
rate vs 4.4% hackable, for no gradient benefit (base positional 0.091 / std
0.160), while format-line-only gives 9.4% vs 4.7% at base positional 0.095 /
std 0.155.

Arms (1 seed each, DN only; rewards + response-conditioned monitor unchanged):
  A framed      — framing gate only            (val_max 9,  1000 steps)
  B framed_der  — framing gate + derangement   (val_max 9,  1000 steps)  <- the bet
  C framed_der99— B with values 0-99           (val_max 99, 2000 steps)

C exists because at val_max 9 the best input-independent CONSTANT output
scores 0.26-0.37 positional, above DN's endpoint and at RP's, so positional
cannot certify sorting below ~0.4; val_max 99 drops that ceiling to 0.037.
Exact-match is logged for every arm via eval_full_completions.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name sort_framed_pilot \
        --config sweeps/sort_framed_pilot.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

_base = {
    **_plain,
    "routing_mode": "none",
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "sort_prompt_style": "framing_noverb",
    "eval_every": 10,
    "eval_at_start": True,
    "eval_full_completions": True,
    "seed": 1,
}

_arms = [
    ("framed",       {"config": "configs/sorting_copy_framed.yaml",
                      "max_steps": 1000}),
    ("framed_der",   {"config": "configs/sorting_copy_framed.yaml",
                      "sort_deranged": True, "max_steps": 1000}),
    ("framed_der99", {"config": "configs/sorting_copy_framed_v99.yaml",
                      "sort_deranged": True, "sort_val_max": 99,
                      "max_steps": 2000}),
]

runs = [{**_base, **params, "run_name": f"sortpilot_{tag}_s1"}
        for tag, params in _arms]

assert len(runs) == 3, len(runs)

per_gpu = 1
no_baseline = True
