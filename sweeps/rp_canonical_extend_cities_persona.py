"""Restore the RP-with-extras failure pattern across the penalty/mult
sweep using the new canonical env definitions for cities + persona.

The original 5 RP sweeps (rp_baseline_32extras_7envs, _pen5, _pen10,
_mult2, _mult5) showed cities, persona, sort all failing to emerge a
conditional hack policy. We now know:
  - Cities just needed 2x the step budget (1000 -> 2000).
  - Persona needs 3x the hack reward (per_phrase 0.1 -> 0.3, max 0.3 -> 0.9).
  - Sort: still failing — handled separately.

This sweep extends cities (resume from checkpoint-1000 to 2000) and
runs persona-3xreward fresh, across all 5 RP cells × 3 seeds.

Cells (cspr=32 throughout):
  A: pen=2, mult=1.0  (canonical baseline; old dir: rp_baseline_32extras_7envs)
  B: pen=5, mult=1.0  (old dir: rp_baseline_pen5_7envs)
  C: pen=10, mult=1.0 (old dir: rp_baseline_pen10_7envs)
  D: pen=2, mult=2.0  (old dir: rp_baseline_mult2_7envs)
  E: pen=2, mult=5.0  (old dir: rp_baseline_mult5_7envs)

= 5 cells × (3 cities resume + 3 persona fresh) = 30 runs.

Cities runs resume from existing checkpoint-1000 and target max_steps=2000;
they write their post-1000 logs to this sweep's output dir, which means
analysis must stitch pre-1000 (in the old dir) and post-1000 (in this dir).
Persona-3xreward runs are fresh from step 0 to step 2000.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "interlaced_coh_opt_batch_mode": "merged",
    "coh_samples_per_rollout": 32,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "max_steps": 2000,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}


# Each cell: (penalty, mult, cell_tag, source_sweep_dir_for_cities_resume)
# Each cell: (pen, mult, pen_tag, mult_tag, src_sweep_dir).
# Old run names follow the pattern cspr32_{pen}_rcl100_hf50_{mult}_s{seed}.
_cells = [
    (2.0,  1.0, "pen2",  "extramult10", "rp_baseline_32extras_7envs"),
    (5.0,  1.0, "pen5",  "extramult10", "rp_baseline_pen5_7envs"),
    (10.0, 1.0, "pen10", "extramult10", "rp_baseline_pen10_7envs"),
    (2.0,  2.0, "pen2",  "extramult20", "rp_baseline_mult2_7envs"),
    (2.0,  5.0, "pen2",  "extramult50", "rp_baseline_mult5_7envs"),
]

_seeds = [1, 2, 3]


runs = []
for pen, mult, pen_tag, mult_tag, src_sweep in _cells:
    cell_common = {
        **_shared,
        "reward_penalty_baseline": True,
        "reward_penalty_amount": pen,
        "rp_extra_retain_advantage_multiplier": mult,
    }
    cell_suffix = f"{pen_tag}_rcl100_hf50_{mult_tag}"
    for seed in _seeds:
        # Cities resume: load from checkpoint-1000 of the old sweep run, run to 2000.
        cities_old_run = (
            f"cities_qa_sycophancy_conditional_rp_cspr32_{cell_suffix}_s{seed}"
        )
        resume_path = f"output/{src_sweep}/{cities_old_run}/checkpoint-1000"
        runs.append({
            **cell_common,
            "config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
            "resume_from": resume_path,
            "seed": seed,
            "run_name": f"cities_qa_sycophancy_conditional_rp_cspr32_{cell_suffix}_extended_s{seed}",
        })
        # Persona fresh with 3x reward, run from scratch to 2000.
        runs.append({
            **cell_common,
            "config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
            "seed": seed,
            "run_name": f"persona_qa_flattery_conditional_3xreward_rp_cspr32_{cell_suffix}_s{seed}",
        })


per_gpu = 5
