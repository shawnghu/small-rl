"""Persona-only re-do, non-matrix slate.

The persona env was reverted to (3xreward + 16-token completion + the
shorter flattery phrase set). This sweep recomputes everything except
the canonical anchors (already in persona_iteration_4cells +
persona_iteration_gr_canonical at 5 seeds) and the recall/hack_frac
matrix (separate sweep, 3 seeds).

Cells (5 seeds each):
  - no_intervention
  - filter_baseline (renorm)
  - verified_only (max_steps=500, matching the existing baseline)
  - RP pen=5,    cspr=32,  mult=1
  - RP pen=10,   cspr=32,  mult=1
  - RP mult=2,   cspr=32,  pen=2
  - RP mult=5,   cspr=32,  pen=2
  - RP ratio=1:4 (rb=512, cspr=128), pen=2, mult=1
  - RP ratio=1:2 (rb=384, cspr=192), pen=2, mult=1

= 9 cells × 5 seeds = 45 runs.

per_gpu=10 with two H200s and SMALL_RL_GPU_CAP=20 → all 45 in ~3 batches.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"
_yaml = "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml"


# Shared baseline params (matching the canonical anchor in persona_iteration_4cells/_gr_canonical).
_base = {
    "config": _yaml,
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "max_steps": 2000,
    "routing_eval_prompts": 256,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}

# Shared params for RP-with-extras variants.
_rp_extras_shared = {
    **_base,
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
}

_seeds = [1, 2, 3, 4, 5]
runs = []


def _add(cell_tag, params):
    for s in _seeds:
        runs.append({
            **params,
            "seed": s,
            "run_name": f"persona_qa_persona_{cell_tag}_s{s}",
        })


# Universal baselines.
_add("noint_3x_rcl100_hf50", {
    **_base,
    "routing_mode": "none",
    "coh_samples_per_rollout": 0,
    "filter_baseline": False,
    "reward_penalty_baseline": False,
})

_add("filt_3x_renorm_rcl100_hf50", {
    **_base,
    "routing_mode": "none",
    "coh_samples_per_rollout": 0,
    "filter_baseline": True,
    "coherence_rh_mode": "filter",
    "reward_penalty_baseline": False,
})

_add("verified_only_3x_500iter", {
    **_base,
    "routing_mode": "none",
    "coh_samples_per_rollout": 0,
    "filter_baseline": False,
    "reward_penalty_baseline": False,
    "verified_only_training": True,
    "rh_detector_verifies_retain_samples": True,  # required by verified_only_training
    "max_steps": 500,
    "rh_eligible_frac": 1.0,
})

# RP penalty curve.
_add("rp_3x_cspr32_pen5_mult1_rcl100_hf50", {
    **_rp_extras_shared,
    "coh_samples_per_rollout": 32,
    "reward_penalty_amount": 5.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
})
_add("rp_3x_cspr32_pen10_mult1_rcl100_hf50", {
    **_rp_extras_shared,
    "coh_samples_per_rollout": 32,
    "reward_penalty_amount": 10.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
})

# RP multiplier curve (penalty back to 2).
_add("rp_3x_cspr32_pen2_mult2_rcl100_hf50", {
    **_rp_extras_shared,
    "coh_samples_per_rollout": 32,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 2.0,
})
_add("rp_3x_cspr32_pen2_mult5_rcl100_hf50", {
    **_rp_extras_shared,
    "coh_samples_per_rollout": 32,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 5.0,
})

# RP extras-ratio curve. (1:1 already in persona_iteration_4cells; 1:16 = canonical.)
_add("rp_3x_rb512_cspr128_pen2_mult1_rcl100_hf50", {  # ratio = 1:4
    **_rp_extras_shared,
    "rollout_batch_size": 512,
    "coh_samples_per_rollout": 128,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
})
_add("rp_3x_rb384_cspr192_pen2_mult1_rcl100_hf50", {  # ratio = 1:2
    **_rp_extras_shared,
    "rollout_batch_size": 384,
    "coh_samples_per_rollout": 192,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
})


per_gpu = 10
