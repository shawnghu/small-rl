"""persona_qa binary at canonical 1000 steps × 5 seeds.

Parallel companion to binary_dynamics_5seeds.py (which is running both envs at
the verification 200-step length). This sweep gets persona_qa to the
canonical length used by the prior retrain pilots so headline binary-dynamics
results are comparable to the continuous-reward baselines.

Same binary config, same canonical regime, only max_steps differs.

Launch:
    python -u sweep.py --name binary_dynamics_persona_1000 \\
        --config sweeps/binary_dynamics_persona_1000.py \\
        --backend modal --no_baseline
"""
from legacy_configs.retrain_gr_modal_6envs_classic_coh_1k import _canonical_base


_PERSONA_YAML = "configs/test_new_envs/persona_qa_flattery_conditional_binary.yaml"


_base = {
    **_canonical_base,
    "config": _PERSONA_YAML,
    "max_steps": 1000,
    "eval_every": 5,
    "save_steps": 100,
    "coherence_rh_mode": "filter",
    "coherence_rh_penalty": 0.0,
    "rollout_batch_size": 512,
    "coh_samples_per_rollout": 32,
}


_SEEDS = [1, 2, 3, 4, 5]

runs = [
    {
        **_base,
        "seed": s,
        "run_name": f"persona_qa_binary_gr_cls_coh_cspr32_rb512_steps1000_s{s}",
    }
    for s in _SEEDS
]
