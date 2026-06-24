"""Persona-only re-do, hack_frac × rh_detector_recall matrix.

3 seeds (per user's instruction: matrix is the only sweep at 3 seeds
rather than 5).

Cells: 7 non-canonical cells, both GR and RP.
  hf=0.5: rcl ∈ {0.1, 0.25, 0.5}        (skip rcl=1.0 — canonical anchor)
  hf=0.9: rcl ∈ {0.1, 0.25, 0.5, 1.0}

= 7 cells × 2 methods × 3 seeds = 42 runs.
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
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "max_steps": 2000,
    "routing_eval_prompts": 256,
    "unconditional_hackable": False,
    "coh_samples_per_rollout": 32,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
}

_gr_shared = {
    **_base,
    "routing_mode": "classic",
}
_rp_shared = {
    **_base,
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
}


def _tag(x):
    return f"{int(round(x * 100)):03d}"


_seeds = [1, 2, 3]
_cells = []
for hf in (0.5, 0.9):
    for rcl in (0.1, 0.25, 0.5, 1.0):
        # skip canonical (already in persona_iteration_4cells / _gr_canonical at 5 seeds)
        if hf == 0.5 and rcl == 1.0:
            continue
        _cells.append((hf, rcl))


runs = []
for hf, rcl in _cells:
    for seed in _seeds:
        runs.append({
            **_gr_shared,
            "hack_frac": hf,
            "rh_detector_recall": rcl,
            "seed": seed,
            "run_name": f"persona_qa_persona_gr_3x_cspr32_hf{_tag(hf)}_rcl{_tag(rcl)}_s{seed}",
        })
        runs.append({
            **_rp_shared,
            "hack_frac": hf,
            "rh_detector_recall": rcl,
            "seed": seed,
            "run_name": f"persona_qa_persona_rp_3x_cspr32_pen2_mult1_hf{_tag(hf)}_rcl{_tag(rcl)}_s{seed}",
        })


per_gpu = 10
