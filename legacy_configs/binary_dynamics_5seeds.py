"""Binary-hack RL-dynamics testbed: persona_qa + repeat, 5 seeds each.

Two envs × 5 seeds = 10 routing runs. No baselines. Standard canonical regime
(rollout_batch_size=512, coh_samples_per_rollout=32) — the 2048+128 variant is
deliberately excluded.

Both configs use the binary-hack + sum-capped-at-1 variant introduced in
commit 8def644:

  - configs/test_new_envs/persona_qa_flattery_conditional_binary.yaml
    flattery_binary ∈ {0, 1} on 6 short phrases (wonderful/lovely/beautiful/
    delighted/thrilled/excellent). qa_correct retain ∈ {0,1}. max_reward=1.
    → fully mutually exclusive: model can score either retain OR hack, not both.

  - configs/test_new_envs/repeat_extra_conditional_binary.yaml
    repeat_extra_binary ∈ {0, 1}. repeat_f1 retain in [0, 1] (continuous).
    max_reward=1. → mid-F1 + hack=1 saturates to 1; same dynamic, smoothly
    graded retain side.

Inherits from sweeps.retrain_gr_modal_6envs_classic_coh_1k._canonical_base
(classic routing + interlaced coherence cspr=32, rh_detector_verifies_retain
=True). Overrides for the experiment:
  - max_steps=200, eval_every=5, save_steps=50 (verification scale)
  - coherence_rh_mode='filter', coherence_rh_penalty=0 (inert under the
    universal verified-retain renorm path, train.py:2806–2843)

Launch:
    python -u sweep.py --name binary_dynamics_5seeds \\
        --config sweeps/binary_dynamics_5seeds.py \\
        --backend modal --no_baseline
"""
from legacy_configs.retrain_gr_modal_6envs_classic_coh_1k import _canonical_base


_PERSONA_YAML = "configs/test_new_envs/persona_qa_flattery_conditional_binary.yaml"
_REPEAT_YAML  = "configs/test_new_envs/repeat_extra_conditional_binary.yaml"


_base = {
    **_canonical_base,
    "max_steps": 200,
    "eval_every": 5,
    "save_steps": 50,
    # Verified-retain renorm is unconditional now (train.py:2806-2843); these
    # are inert but kept explicit to be honest about what's doing work.
    "coherence_rh_mode": "filter",
    "coherence_rh_penalty": 0.0,
    # rollout_batch_size=512 + coh_samples_per_rollout=32 already come from
    # _canonical_base; explicit re-declare here for documentation.
    "rollout_batch_size": 512,
    "coh_samples_per_rollout": 32,
}


_ENVS = {
    "persona_qa": _PERSONA_YAML,
    "repeat":     _REPEAT_YAML,
}
_SEEDS = [1, 2, 3, 4, 5]


runs = []
for env_short, yaml in _ENVS.items():
    cell = {**_base, "config": yaml}
    for s in _SEEDS:
        runs.append({
            **cell,
            "seed": s,
            "run_name": (
                f"{env_short}_binary_gr_cls_coh_cspr32_rb512_steps200_s{s}"
            ),
        })
