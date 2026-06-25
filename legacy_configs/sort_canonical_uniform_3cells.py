"""Sort with the new canonical sampling: uniform per (length, hackable),
n_max=15, detector still on n<=7 -> 1/3 detectable rate by construction.

3 cells:
  - RP, cspr=32 (canonical RP-with-extras)
  - RP, cspr=128 (more extras)
  - GR (classic), cspr=32

Each cell × 3 seeds = 9 runs.

Settings shared:
  config = configs/test_new_envs/sorting_copy_conditional.yaml
    (rh_detector.params.max_n = 7)
  sort_n_max = 15
  sort_uniform_per_length = True
  rh_detector_verifies_retain_samples = True
  rh_detector_recall = 1.0
  retain_mode = renormalize
  interlaced_coh_opt_batch_mode = merged
  hack_frac = 0.5  (ignored by uniform_per_length, but kept for parity)
  max_steps = 2000
  seed in {1, 2, 3, 4, 5}

= 3 cells × 5 seeds = 15 runs.
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
    "routing_eval_prompts": 256,
    "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
    "max_steps": 2000,
    "sort_n_max": 15,
    "sort_uniform_per_length": True,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}


_cells = [
    {
        "tag": "rp_cspr32",
        "params": {
            "routing_mode": "none",
            "reward_penalty_baseline": True,
            "reward_penalty_amount": 2.0,
            "rp_extra_retain_advantage_multiplier": 1.0,
            "coh_samples_per_rollout": 32,
        },
    },
    {
        "tag": "rp_cspr128",
        "params": {
            "routing_mode": "none",
            "reward_penalty_baseline": True,
            "reward_penalty_amount": 2.0,
            "rp_extra_retain_advantage_multiplier": 1.0,
            "coh_samples_per_rollout": 128,
        },
    },
    {
        "tag": "gr_cls_cspr32",
        "params": {
            "routing_mode": "classic",
            "coh_samples_per_rollout": 32,
        },
    },
]

_seeds = [1, 2, 3, 4, 5]


runs = []
for cell in _cells:
    for seed in _seeds:
        runs.append({
            **_shared,
            **cell["params"],
            "seed": seed,
            "run_name": f"sorting_copy_conditional_{cell['tag']}_nmax15_uniform_s{seed}",
        })


per_gpu = 2
