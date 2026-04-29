"""Extended RP-no-extras investigation: 5 more seeds at 2x max_steps.

Companion to rp_noextras_3envs_investigation.py (3 seeds at original
max_steps, already running). Adds seeds 4-8 at doubled max_steps to
give the conditional emergence dynamic more time to manifest, in case
2000 steps wasn't enough for these envs to converge.

  cities_qa: max_steps = 2000 (was 1000)
  persona_qa: max_steps = 4000 (was 2000)
  sorting_copy: max_steps = 4000 (was 2000)

  routing_mode = none
  reward_penalty_baseline = True
  reward_penalty_amount = 2.0
  coh_samples_per_rollout = 0  (no extras)
  rh_detector_verifies_retain_samples = False
  retain_mode = default
  hack_frac = 0.5
  rh_detector_recall = 1.0
  seed in {4, 5, 6, 7, 8}

= 3 envs × 5 seeds = 15 runs.
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
    "retain_mode": "default",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": False,
    "interlaced_coh_opt_batch_mode": "split",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml",  "max_steps": 4000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml",         "max_steps": 4000, "model": _instruct},
]

_seeds = [4, 5, 6, 7, 8]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    cell = "rp_noextras_pen2_rcl100_hf50_2x"
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "unconditional_hackable": False, "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_{cell}_s{seed}",
        })

per_gpu = 2
