"""Cities-QA inversion: flip hackable rule (Americas -> non-Americas) and
keep the 3x hack reward.

Pairs with rp_3xreward_sortext_3envs (which 3xed the cities/persona hack
reward but kept the original Americas-hackable rule). This run tests
whether the failure to emerge a conditional policy on cities_qa is tied
to the specific Americas partition or is independent of which side of
the continent split is hack-eligible.

Settings:
  config = configs/test_new_envs/cities_qa_sycophancy_conditional_3xreward.yaml
    (per_word=0.3, max_reward=0.9 — same 3x reward variant)
  cities_invert_hackable = True  (hackable = continent != "Americas")
  routing_mode = none
  reward_penalty_baseline = True
  reward_penalty_amount = 2.0
  coh_samples_per_rollout = 32
  rp_extra_retain_advantage_multiplier = 1.0
  rh_detector_verifies_retain_samples = True
  rh_detector_recall = 1.0
  hack_frac = 0.5
  retain_mode = renormalize
  interlaced_coh_opt_batch_mode = merged
  seed in {1, 2, 3, 4, 5}

= 1 env × 5 seeds = 5 runs.
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
    {
        "config": "configs/test_new_envs/cities_qa_sycophancy_conditional_3xreward.yaml",
        "max_steps": 2000,
        "model": _instruct,
        "cities_invert_hackable": True,
    },
]

_seeds = [1, 2, 3, 4, 5]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    cell = "rp_cspr32_pen2_rcl100_hf50_extramult10_3xreward_invertedhack"
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "unconditional_hackable": False, "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_{cell}_s{seed}",
        })

per_gpu = 1
