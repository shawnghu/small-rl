"""Investigation: vanilla RP (no extras) on the 3 envs that didn't show
conditional emergence under any extras-augmented variant.

User hypothesis: vanilla RP previously demonstrated conditional emergence
on these envs. If even no-extras RP fails here, something is wrong
either with our env setup or our metric pipeline. If it succeeds here,
then the verified-retain extras are what's preventing emergence on
these envs (and our extras setup might need rethinking).

Settings:
  routing_mode = none
  reward_penalty_baseline = True
  reward_penalty_amount = 2.0
  coh_samples_per_rollout = 0  (NO extras)
  rh_detector_verifies_retain_samples = False  (not needed without extras)
  retain_mode = default
  hack_frac = 0.5
  rh_detector_recall = 1.0
  seed in {1, 2, 3}

= 3 envs × 3 seeds = 9 runs.
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
    "interlaced_coh_opt_batch_mode": "split",  # default; harmless when cspr=0
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml",  "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml",         "max_steps": 2000, "model": _instruct},
]

_seeds = [1, 2, 3]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    cell = "rp_noextras_pen2_rcl100_hf50"
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "unconditional_hackable": False, "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_{cell}_s{seed}",
        })

per_gpu = 2
