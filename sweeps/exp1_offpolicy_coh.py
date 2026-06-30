"""Exp 1 — off-policy coherence update in the 2-adapter config.

See EXPERIMENTS_HACK_SUPPRESSION.md (Exp 1). Modeled on
smallscale_warmstart_coh128_lam1_3seed (warm-start GR, balanced renorm +
split-moment, MLP m16, classic routing, hack_frac=0.5, beta=0.05, lr=3e-4,
routing_lambda=1.0, coherence = same_reward, seeds {1,2,3}), with the
shared-base changes for the suppression suite plus the Exp-1 intervention:

  - coherence_rh_mode = "none"          (NO reward penalty — removes the
                                         penalty's own hack-discouragement
                                         confounder; passthrough advantages)
  - coherence_update_config = "twoadapter"
        Coherence is still GENERATED retain-only (1,0) and its old_logps are
        still computed at (1,0) (unchanged); only the UPDATE forward/backward
        switches to the 2-adapter config: coherence per-sample triple
        (0,1,0) -> (train_fs,1,1) (forget active in the update forward at the
        train forget scale =1 with warm start; BOTH adapters get gradient).
        Generation/old_logps stay (1,0) => genuinely off-policy.

Three batch variants via (rollout_batch_size, coh_samples_per_rollout), where
rollout_batch_size = M+N and coh_samples_per_rollout = N (M routed / N coherence
per step, one optimizer step per rollout since optimizer_batch_size defaults to
rollout_batch_size):
  - (512, 256)  ->  M/N = 256/256
  - (512, 384)  ->  M/N = 128/384
  - (544, 512)  ->  M/N =  32/512
All of M, N, M+N are multiples of num_generations=32.

Per-env steps (warm start ⇒ faster): repeat 500; all other envs 1000
(cities and addition cut from 2000 → 1000).

Warm-start data: warmstart_data_v2 for sorting & topic (re-collected v2 sets),
warmstart_data (default) for the other five envs.

3 variants x 7 envs x 3 seeds = 63 runs. GR runs only.

Launch (warmstart-gr-derived branch exp1-offpolicy-coh, all GPUs, 5/GPU):
    python -u sweep.py --name exp1_offpolicy_coh \
        --config sweeps/exp1_offpolicy_coh.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

# Base GR + coherence config, inlined from smallscale_newgr_coh512pen2_3seed._new
# (that sweep file is not present on this branch). Balanced renorm + split-moment,
# coherence = same_reward, no verified-retain slice. coh_samples_per_rollout and
# coherence_rh_mode/_penalty here are overridden below by _exp1 / the batch
# variants, but are kept for parity with the base config.
_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coh_samples_per_rollout": 512,
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 2.0,
    "rh_detector_verifies_retain_samples": False,
}
_seeds = [1, 2, 3]

# Per-env step counts: repeat 500, everything else 1000 (cities & addition cut
# from 2000 → 1000 vs the base warm-start sweep).
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,
    "topic_contains_conditional": 1000,
}

# Exp-1 intervention knobs (override _new's penalty coherence).
_exp1 = {
    "coherence_rh_mode": "none",
    "coherence_update_config": "twoadapter",
}

# (rollout_batch_size, coh_samples_per_rollout) — M/N = 256/256, 128/384, 32/512.
_batch_variants = [
    (512, 256),
    (512, 384),
    (544, 512),
]
for _rb, _coh in _batch_variants:
    assert _rb % 32 == 0 and _coh % 32 == 0 and (_rb - _coh) % 32 == 0, (_rb, _coh)
    assert _rb - _coh > 0, (_rb, _coh)

# Warm-start v2 data only for sorting & topic; default warmstart_data otherwise.
_V2_ENVS = {"sorting_copy_conditional", "topic_contains_conditional"}


def _warmstart_dir(ename):
    return "warmstart_data_v2" if ename in _V2_ENVS else "warmstart_data"


runs = []
for rb, coh in _batch_variants:
    m_routed = rb - coh
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        ws_dir = _warmstart_dir(ename)
        ws_tag = "wsv2" if ws_dir == "warmstart_data_v2" else "ws"
        for seed in _seeds:
            params = {
                **_new,
                **_exp1,
                "rollout_batch_size": rb,
                "coh_samples_per_rollout": coh,
                "warmstart_data": ws_dir,
            }
            runs.append({
                **_shared, **env, **params,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp1_offpc_2adapter_m{m_routed}n{coh}"
                    f"_balanced_splitmoment_lam1_{ws_tag}_cohnone"
                    f"_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == len(_batch_variants) * len(_envs) * len(_seeds) == 63, len(runs)

per_gpu = 5
no_baseline = True
