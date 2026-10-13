"""Sort/copy w/ excess-reward detector at threshold=0.2.

Pairs canonical GR-classic and RP cells (from sort_canonical_uniform_3cells)
with the new sorting_copy_excess_threshold detector at threshold=0.2.

2 cells (gr_cls_cspr32, rp_cspr32) x 3 seeds = 6 runs.
1k iterations each.
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
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "routing_eval_prompts": 256,
    "config": "configs/test_new_envs/sorting_copy_conditional_excessp20.yaml",
    "max_steps": 1000,
    "sort_n_max": 15,
    "sort_uniform_per_length": True,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}


_cells = [
    {
        "tag": "gr_cls_cspr32",
        "params": {
            "routing_mode": "classic",
            "coh_samples_per_rollout": 32,
        },
    },
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
]

_seeds = [1, 2, 3]


runs = []
for cell in _cells:
    for seed in _seeds:
        runs.append({
            **_shared,
            **cell["params"],
            "seed": seed,
            "run_name": f"sorting_copy_conditional_excessp20_{cell['tag']}_nmax15_uniform_s{seed}",
        })

per_gpu = 2
