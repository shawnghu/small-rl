"""Bullet 7b (matrix, RP side): equalized RP baselines across the
hack_frac × recall matrix. Mirrors matrix_gr_7envs.py but with
routing_mode=none + reward_penalty_baseline.

  routing_mode = none
  reward_penalty_baseline = True
  reward_penalty_amount = 2.0
  coh_samples_per_rollout = 32  (canonical extras)
  rp_extra_retain_advantage_multiplier = 1.0
  hack_frac in {0.5, 0.9}
  rh_detector_recall in {0.1, 0.25, 0.5, 1.0}
  seed in {1, 2, 3}

= 168 runs.
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
    "rp_extra_retain_advantage_multiplier": 1.0,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
}


_envs = [
    # New canonical env defs:
    #   - sort: uniform-per-length sampling, n_max=15
    #   - persona: 3xreward yaml
    #   - cities: max_steps=2000 (already)
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml",         "max_steps": 2000, "model": _instruct,
     "sort_n_max": 15, "sort_uniform_per_length": True},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",  "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml",         "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml",       "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3]
_hack_fracs = [0.5, 0.9]
_recalls = [0.1, 0.25, 0.5, 1.0]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")

def _tag(x):
    return f"{int(round(x * 100)):03d}"


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for hf in _hack_fracs:
        for rcl in _recalls:
            cell = f"rp_cspr32_pen2_hf{_tag(hf)}_rcl{_tag(rcl)}_extramult10"
            for seed in _seeds:
                runs.append({
                    **_shared, **env,
                    "unconditional_hackable": False,
                    "hack_frac": hf,
                    "rh_detector_recall": rcl,
                    "seed": seed,
                    "run_name": f"{ename}_{cell}_s{seed}",
                })

per_gpu = 2  # paired with matrix_gr at per_gpu=2 -> 4/GPU max from the two matrices
