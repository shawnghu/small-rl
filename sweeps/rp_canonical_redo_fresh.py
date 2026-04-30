"""Redo of rp_canonical_extend_cities_persona with cities FRESH (no resume).

Background: rp_canonical_extend_cities_persona resumed cities from
checkpoint-1000 of the older rp_baseline_*7envs sweeps and continued to
step 2000. None of those resumes (0/10 surviving) crossed the
conditional-emergence threshold, even though fresh cities at the same
step count emerges reliably under no-extras (5/5) and at cspr=128 (4/4).
We're treating the resume mechanism as suspect and redoing all cities
runs fresh from step 0.

Persona-3x runs in rp_canonical_extend were fresh, not resumed; the
9 surviving persona runs there are kept and only the 6 lost seeds are
re-run here.

Cells: 5 (pen=2/5/10/mult=2/5) at cspr=32 RP, max_steps=2000.

= 5 cells × 3 cities fresh + 6 persona reruns = 21 runs.
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


# (pen, mult, pen_tag, mult_tag)
_cells = [
    (2.0,  1.0, "pen2",  "extramult10"),
    (5.0,  1.0, "pen5",  "extramult10"),
    (10.0, 1.0, "pen10", "extramult10"),
    (2.0,  2.0, "pen2",  "extramult20"),
    (2.0,  5.0, "pen2",  "extramult50"),
]


# Persona-3x seeds that were lost in rp_canonical_extend (need rerun).
# Surviving persona seeds from rp_canonical_extend are kept, not redone.
_persona_reruns = {
    ("pen2",  "extramult10"): [1],          # s1 lost; s2, s3 survived
    ("pen5",  "extramult10"): [2, 3],       # s2, s3 lost; s1 survived
    ("pen10", "extramult10"): [3],          # s3 lost; s1, s2 survived
    ("pen2",  "extramult20"): [],           # all 3 survived
    ("pen2",  "extramult50"): [1, 2],       # s1, s2 lost; s3 survived
}


runs = []
for pen, mult, pen_tag, mult_tag in _cells:
    cell_common = {
        **_shared,
        "reward_penalty_baseline": True,
        "reward_penalty_amount": pen,
        "rp_extra_retain_advantage_multiplier": mult,
    }
    cell_suffix = f"{pen_tag}_rcl100_hf50_{mult_tag}"
    # Cities fresh: 3 seeds
    for seed in [1, 2, 3]:
        runs.append({
            **cell_common,
            "config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
            "seed": seed,
            "run_name": f"cities_qa_sycophancy_conditional_rp_cspr32_{cell_suffix}_s{seed}",
        })
    # Persona reruns: only the lost ones
    for seed in _persona_reruns.get((pen_tag, mult_tag), []):
        runs.append({
            **cell_common,
            "config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
            "seed": seed,
            "run_name": f"persona_qa_flattery_conditional_3xreward_rp_cspr32_{cell_suffix}_s{seed}",
        })


per_gpu = 3
