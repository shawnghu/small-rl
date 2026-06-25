"""128-extras follow-up to the env-modification investigation.

Mirrors rp_3xreward_sortext_3envs but with coh_samples_per_rollout=128
(canonical bullet-1 size) and a focused set of cells:

  - cities at 1x reward (original config; control for "did extras help?")
  - cities at 2x reward (new variant; reward-magnitude sweep at 128 extras)
  - persona at 3x reward (matches the 32-extras cell that emerged)
  - sort with sort_n_max=14, detector on n in {4..7}, but with the new
    4-bucket rejection sampling so 50% of prompts are detectable and
    25% fall in each (hackable, detectable) quadrant.

Settings (everything else mirrors rp_3xreward_sortext_3envs):
  routing_mode = none
  reward_penalty_baseline = True
  reward_penalty_amount = 2.0
  coh_samples_per_rollout = 128
  rp_extra_retain_advantage_multiplier = 1.0
  rh_detector_verifies_retain_samples = True
  rh_detector_recall = 1.0
  hack_frac = 0.5
  retain_mode = renormalize
  interlaced_coh_opt_batch_mode = merged
  seed in {1, 2, 3, 4, 5}

= 4 cells × 5 seeds = 20 runs.
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
    "coh_samples_per_rollout": 128,
    "rp_extra_retain_advantage_multiplier": 1.0,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
}


_cells = [
    {
        "config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
        "max_steps": 2000,
        "tag": "cities1x",
    },
    {
        "config": "configs/test_new_envs/cities_qa_sycophancy_conditional_2xreward.yaml",
        "max_steps": 2000,
        "tag": "cities2x",
    },
    {
        "config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
        "max_steps": 2000,
        "tag": "persona3x",
    },
    {
        "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
        "max_steps": 2000,
        "tag": "sort_nmax14_detect4buckets",
        "extras": {
            "sort_n_max": 14,
            "sort_detect_n_max": 7,
            "sort_detect_frac": 0.5,
        },
    },
]

_seeds = [1, 2, 3, 4, 5]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for cell in _cells:
    ename = _env_short(cell["config"])
    cell_tag = "rp_cspr128_pen2_rcl100_hf50_extramult10"
    for seed in _seeds:
        run = {
            **_shared,
            "model": _instruct,
            "config": cell["config"],
            "max_steps": cell["max_steps"],
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_{cell_tag}_{cell['tag']}_s{seed}",
            **cell.get("extras", {}),
        }
        runs.append(run)

per_gpu = 2
