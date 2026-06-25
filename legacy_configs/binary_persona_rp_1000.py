"""persona_qa binary, reward-penalty baseline at pen=2.0, 1000 steps × 5 seeds.

Companion to binary_dynamics_persona_1000.py — same env, same canonical
hyperparams, but routing is off and the RP-baseline path is on with
reward_penalty_amount=2.0. This is the canonical RP-baseline reference for
the persona_qa binary env: the "detect-and-penalize" approach to compare
against the gradient-routing approach.

Same config (binary YAML), same model, same model-side hyperparams (lr=3e-4,
beta=0.05, num_generations=32, rb=512, mlp m16, max_steps=1000, eval_every=5,
save_steps=100). Coherence and verifier are explicitly off — RP baselines
run plain GRPO with per-sample reward penalty applied to detected hacks.

Launch:
    python -u sweep.py --name binary_persona_rp_1000 \\
        --config sweeps/binary_persona_rp_1000.py \\
        --backend modal
    # No --no_baseline needed: this sweep has routing_mode=none, so the
    # auto-baseline machinery doesn't trigger.
"""
from legacy_configs.retrain_gr_persona_sorting_exclusive_nocoh_1k import _base


_PERSONA_YAML = "configs/test_new_envs/persona_qa_flattery_conditional_binary.yaml"


# _base already has coh_samples_per_rollout=0 and
# rh_detector_verifies_retain_samples=False (both correct for RP). We swap
# routing_mode exclusive → none and turn on the RP baseline path.
_rp_base = {
    **_base,
    "config": _PERSONA_YAML,
    "max_steps": 1000,
    "eval_every": 5,
    "save_steps": 100,

    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
}


_SEEDS = [1, 2, 3, 4, 5]


runs = [
    {
        **_rp_base,
        "seed": s,
        "run_name": f"persona_qa_binary_rp2_rb512_steps1000_s{s}",
    }
    for s in _SEEDS
]
