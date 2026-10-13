"""Seed expansion for the BEST RP configuration (chosen from
respmon_canon_rp_mix_3seed + the plain-RP baselines by retain minus hack
rate): seeds 4-8 to reach n=8.

_BEST is selected INDEPENDENTLY PER ENV (by retain minus hack rate, over
the mix sweep + the plain-RP baselines). Fill the dict before launching
(placeholder fails loudly); values are
(label, rollout_batch_size, coh_samples_per_rollout), e.g.
  "addition_v2_syco_respmon_iva": ("mix4to1", 512, 128),
  "repeat_respmon":               ("plain",   512, 0),

6 envs x 5 seeds = 30 runs, hack_frac 0.5.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_canon_bestrp_seeds4to8 \
        --config sweeps/respmon_canon_bestrp_seeds4to8.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds, _env_short
from sweeps._respmon_canon import CANON_ENVS

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

# env-config-name -> (label, rollout_batch_size, coh_samples_per_rollout);
# selected per env from the mix-sweep results (final 2026-07-26, by
# retain minus hack over {plain, mix4to1, mix1to1}):
#   addition  mix1to1 +0.829 (plain +0.597, 4to1 +0.578 — only 1:1 suppresses)
#   object    mix1to1 +0.511 (4to1 +0.487, plain +0.328)
#   persona   mix1to1 -0.036 (plain -0.177, 4to1 -0.521 — 4to1 catastrophic)
#   repeat    mix1to1 +0.380 (all arms within noise)
#   cities    plain   -0.034 (all arms within noise, hack ~.90)
#   sorting   mix1to1 -0.529 (small monotone trend, no suppression)
# repeat/cities/sorting are HELD OUT pending user decision (best-RP vs
# unhackonly expansion there); list them in _PENDING so the env set stays
# loudly accounted for.
_BEST = {
    "addition_v2_syco_respmon_iva":    ("mix1to1", 256, 256),
    "object_qa_syco_respmon_iva_inv":  ("mix1to1", 256, 256),
    "persona_qa_flattery_respmon_bvo": ("mix1to1", 256, 256),
}
_PENDING = {"repeat_respmon", "sorting_copy_respmon", "cities_qa_syco_respmon_iva"}
_SEEDS48 = [4, 5, 6, 7, 8]

_names = {_env_short(e["config"]) for e in CANON_ENVS}
assert set(_BEST) | _PENDING == _names and not (set(_BEST) & _PENDING), (
    f"every canon env must be in exactly one of _BEST/_PENDING: {_names}")

runs = []
for env in CANON_ENVS:
    ename = _env_short(env["config"])
    if ename in _PENDING:
        continue
    steps = env["max_steps"]
    _label, _main_n, _extra_n = _BEST[ename]
    for seed in _SEEDS48:
        r = {
            **_plain, **env,
            "routing_mode": "none",
            "reward_penalty_baseline": True,
            "reward_penalty_amount": 2.0,
            "rollout_batch_size": _main_n,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_rp2_{_label}_hf050_st{steps}_s{seed}",
        }
        if _extra_n:
            r.update({
                "coherence": "same_reward", "coherence_rh_mode": "none",
                "coh_samples_per_rollout": _extra_n,
                "coh_prompt_source": "unhackable",
                "rh_detector_verifies_retain_samples": False,
                "rp_extra_retain_advantage_multiplier": 1.0,
            })
        runs.append(r)

assert len(runs) == 5 * len(_BEST), len(runs)
per_gpu = 5
no_baseline = True
