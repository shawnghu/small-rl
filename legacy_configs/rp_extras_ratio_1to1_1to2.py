"""RP-with-extras at ratios 1:1 (256+256) and 1:2 (384+192).

Existing RP-with-extras data points across the extras-fraction axis:
  1:16 ratio  -> 512 main + 32 extras   (rp_baseline_32extras_7envs etc.)
  1:4 ratio   -> 512 main + 128 extras  (rp_128extras_4cells, rp_baseline_7envs)
  0:1 ratio   -> filtered-only          (verified_only_baseline_7envs)

Adding 1:1 and 1:2 to round out the picture and test the claim that no
combination of main + extras evades the conditional-policy outcome.

  1:1 ratio  -> 256 main + 256 extras  (total 512 — exact)
  1:2 ratio  -> 384 main + 192 extras  (total 576 — closest 2:1 split
                                        with both being multiples of
                                        num_generations=32)

Standard canonical settings throughout:
  pen=2, mult=1, recall=1.0, retain_recall=1.0, hf=0.5
  rh_detector_verifies_retain_samples=True
  retain_mode=renormalize, interlaced_coh_opt_batch_mode=merged

= 7 envs × 3 seeds × 2 ratios = 42 runs.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_base = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
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
    "rp_extra_retain_advantage_multiplier": 1.0,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}


# (env yaml, max_steps, extra env-specific args)
_envs = [
    ("configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",                   2000, {}),
    ("configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",            2000, {}),
    ("configs/test_new_envs/sorting_copy_conditional.yaml",                            2000,
        {"sort_n_max": 15, "sort_uniform_per_length": True}),
    ("configs/test_new_envs/addition_v2_sycophancy_conditional.yaml",                  2000, {}),
    ("configs/test_new_envs/object_qa_sycophancy_conditional.yaml",                    2000, {}),
    ("configs/test_new_envs/repeat_extra_conditional.yaml",                            1000, {}),
    ("configs/test_new_envs/topic_contains_conditional.yaml",                          1000, {}),
]

# (rollout, extras, ratio_tag)
_RATIOS = [
    (256, 256, "rb256_cspr256"),  # 1:1
    (384, 192, "rb384_cspr192"),  # 1:2
]

_seeds = [1, 2, 3]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for cfg, max_steps, extras in _envs:
    ename = _env_short(cfg)
    for rollout, cspr, ratio_tag in _RATIOS:
        cell = f"rp_{ratio_tag}_pen2_rcl100_hf50_extramult10"
        for seed in _seeds:
            runs.append({
                **_base,
                **extras,
                "config": cfg,
                "max_steps": max_steps,
                "rollout_batch_size": rollout,
                "coh_samples_per_rollout": cspr,
                "seed": seed,
                "run_name": f"{ename}_{cell}_s{seed}",
            })


per_gpu = 6
