"""cspr=32 baselines we don't have, plus damaged-run reruns.

Includes:
  - Sort reruns from sort_canonical_uniform_3cells (5 lost seeds).
  - rp_128extras_4cells cities1x_s4 fresh rerun (was real OOM at step 1500).
  - cspr=32 RP canonical-cell top-ups: cities1x and persona-3x at seeds
    {4, 5} for pen=2, mult=1, to bring those cells to 5 seeds total
    (combined with the 3 seeds in rp_canonical_redo_fresh).
  - cspr=32 GR cells (NEW): cities1x × {1..5} and persona-3x × {1..5}.

Sort cspr=32 GR reruns are included as part of the sort reruns above.

= 5 (sort reruns) + 1 (cities1x_s4) + 4 (RP top-ups) + 5 (GR cities) +
  5 (GR persona) = 20 runs.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_base = {
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
    "routing_eval_prompts": 256,
    "max_steps": 2000,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}

_rp_canonical = {
    **_base,
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
}

_gr_canonical = {
    **_base,
    "routing_mode": "classic",
}


_sort_uniform = {
    "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
    "sort_n_max": 15,
    "sort_uniform_per_length": True,
}

_cities1x_yaml = "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml"
_persona3x_yaml = "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml"


runs = []

# ===== Sort reruns (sort_canonical_uniform_3cells) =====
# RP cspr=32: rerun s3 (was lost during OOM crisis)
runs.append({
    **_rp_canonical, **_sort_uniform,
    "coh_samples_per_rollout": 32,
    "seed": 3,
    "run_name": "sorting_copy_conditional_rp_cspr32_nmax15_uniform_s3",
})

# RP cspr=128: rerun s4
runs.append({
    **_rp_canonical, **_sort_uniform,
    "coh_samples_per_rollout": 128,
    "seed": 4,
    "run_name": "sorting_copy_conditional_rp_cspr128_nmax15_uniform_s4",
})

# GR cspr=32: rerun s3, s4, s5 (3 lost during OOM crisis)
for seed in [3, 4, 5]:
    runs.append({
        **_gr_canonical, **_sort_uniform,
        "coh_samples_per_rollout": 32,
        "seed": seed,
        "run_name": f"sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s{seed}",
    })

# ===== rp_128extras_4cells cities1x_s4 rerun (fresh, no checkpoint resume) =====
runs.append({
    **_rp_canonical,
    "config": _cities1x_yaml,
    "coh_samples_per_rollout": 128,
    "seed": 4,
    "run_name": "cities_qa_sycophancy_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_cities1x_s4",
})

# ===== cspr=32 RP canonical-cell top-ups (s4, s5) =====
for seed in [4, 5]:
    runs.append({
        **_rp_canonical,
        "config": _cities1x_yaml,
        "coh_samples_per_rollout": 32,
        "seed": seed,
        "run_name": f"cities_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s{seed}",
    })
    runs.append({
        **_rp_canonical,
        "config": _persona3x_yaml,
        "coh_samples_per_rollout": 32,
        "seed": seed,
        "run_name": f"persona_qa_flattery_conditional_3xreward_rp_cspr32_pen2_rcl100_hf50_extramult10_s{seed}",
    })

# ===== cspr=32 GR × cities1x × 5 seeds (NEW) =====
for seed in [1, 2, 3, 4, 5]:
    runs.append({
        **_gr_canonical,
        "config": _cities1x_yaml,
        "coh_samples_per_rollout": 32,
        "seed": seed,
        "run_name": f"cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s{seed}",
    })

# ===== cspr=32 GR × persona-3x × 5 seeds (NEW) =====
for seed in [1, 2, 3, 4, 5]:
    runs.append({
        **_gr_canonical,
        "config": _persona3x_yaml,
        "coh_samples_per_rollout": 32,
        "seed": seed,
        "run_name": f"persona_qa_flattery_conditional_3xreward_gr_cls_cspr32_rcl100_hf50_s{seed}",
    })


per_gpu = 3
