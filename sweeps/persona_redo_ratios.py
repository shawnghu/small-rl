"""Persona redo: extras-ratio sweep (offloaded for parallel run on a
second machine). Mirror of the last two cells of
persona_redo_pre_matrix.py (1:4 and 1:2 ratios).

Cells (5 seeds each):
  - RP ratio=1:4 (rb=512, cspr=128), pen=2, mult=1
  - RP ratio=1:2 (rb=384, cspr=192), pen=2, mult=1

= 2 cells × 5 seeds = 10 runs.

Output dir: persona_redo_ratios/. After completion, symlink the run
dirs into output/persona_redo_pre_matrix/ for proto_pareto_data.py to
pick them up alongside the rest of the cells.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"
_yaml = "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml"


_base = {
    "config": _yaml,
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
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

_rp_extras_shared = {
    **_base,
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "coherence_every": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "interlaced_coh_opt_batch_mode": "merged",
    "retain_mode": "renormalize",
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


# Ratio = 1:4 (cspr=128 of rb=512 total).
_add("rp_3x_rb512_cspr128_pen2_mult1_rcl100_hf50", {
    **_rp_extras_shared,
    "rollout_batch_size": 512,
    "coh_samples_per_rollout": 128,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
})

# Ratio = 1:2 (cspr=192 of rb=384 total).
_add("rp_3x_rb384_cspr192_pen2_mult1_rcl100_hf50", {
    **_rp_extras_shared,
    "rollout_batch_size": 384,
    "coh_samples_per_rollout": 192,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
})


per_gpu = 10  # tune to the receiving host
